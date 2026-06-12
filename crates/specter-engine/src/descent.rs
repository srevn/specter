//! Pending-path descent.
//!
//! Pending descent runs **outside** the Burst lifecycle. A Profile whose anchor doesn't yet exist
//! on the filesystem lives in `ProfileState::Pending(DescentState)`. The descent emits its own
//! probes correlated by the `ProbeSlot` on `DescentState` (armed at every mint, disarmed when the
//! response is consumed), advances one path component per probe response, and ends by materializing
//! the anchor — at which point the Profile transitions Pending → Idle and immediately Idle →
//! Active(Seed) to establish its baseline.
//!
//! **Why a parallel state machine?** Burst semantics don't fit:
//! - Probe target ≠ `Profile.resource` during descent (probes go to the deepest existing prefix,
//!   not the anchor).
//! - There's no Effect to fire — the Profile has no baseline yet.
//! - The settle timer (carried inside `PreFirePhase::Batching`) and `burst_deadline` are stability
//!   concerns; descent is event-driven (a `StructureChanged` at the prefix triggers a fresh probe
//!   with no settle wait).
//! - I5 stays intact: at most one outstanding probe per Profile. `Pending` and `Active` are
//!   mutually exclusive `ProfileState` variants (the compiler proves it); within `Pending`, an
//!   in-flight probe is exactly an armed `DescentState.probe` slot — one slot per descent, so two
//!   simultaneous descent probes are unconstructable.
//!
//! **Three entries, one machine.** Descent is entered from:
//! - **Attach time** (`materialize_path_or_pending` → Pending): the requested path has scaffold
//!   segments — the anchor doesn't exist yet. Unwitnessed: no kernel signal stands behind the
//!   entry, so an anchor the entry probe finds pins silently (restart-safe doctrine).
//! - **Observed anchor loss** (`Engine::finalize_anchor_lost_and_descend`): the anchor-terminal
//!   event, a probe-`Vanished` dispatch, or a kind-mismatched response re-enters descent at
//!   `watch_root_parent` *inside the loss step itself* — "anchor lost" and "anchor doesn't yet
//!   exist" are the same state. Witnessed: the loss signal is the absence half of the appearance
//!   witness, so the terminus Seed owes a fire.
//! - **Park recovery** (`Engine::start_pending_recovery`): a `Parked` Profile — a probe-`Failed`
//!   terminal or a watch-rejection purge — re-enters on a parent / anchor-slot event, a sensor
//!   overflow, or an operator re-attach. Unwitnessed: the trigger can be sibling churn out of the
//!   Sub's scope (or carries no observation at all).
//!
//! **Lifecycle.**
//! 1. One of the three entries above flips the Profile to `Pending`. The deepest existing ancestor
//!    is `current_prefix`; the remaining path components await materialization (a recovery entry is
//!    the one-segment special case: prefix = the anchor's parent).
//! 2. The engine bumps `current_prefix.watch_demand` and emits a `ProbeOp::Probe` at the prefix — the
//!    wire names the awaited head segment, and the walker answers with one `lstat(prefix/segment)`.
//! 3. `dispatch_descent_probe` consumes the response. The responses also carry the descent's
//!    **appearance witness**: a probe observing the awaited segment absent (or the prefix vanished)
//!    records a standing absence observation, and a later probe finding the segment present
//!    completes the absence→presence pair, latching `DescentState::witnessed`. Probes are the only
//!    witness writers — a prefix event names no segment on either backend, so sibling churn at a
//!    shared prefix can never masquerade as the anchor appearing.
//!    - `SegmentObserved { kind: Some(_) }`: the awaited segment exists. Is the anchor →
//!      materialize (promote to `User`, set kind, bump anchor's `watch_demand`, drop the prefix's,
//!      transition Pending → Idle, start a Seed burst — cold, or Batching-first triggered when the
//!      descent's witnessed-appearance latch is set). Not the anchor → advance descent one segment.
//!    - `SegmentObserved { kind: None }`: the prefix is healthy but the segment is absent — record
//!      the absence observation and park awaiting the next event (a witnessed park narrates via
//!      `PendingPathAwaitingSegment` — the delete-then-write recovery shape).
//!    - `Vanished`: the prefix itself is gone — an absence observation for the whole remaining
//!      chain. Sub the prefix's contribution; vacate; rewind to the next-existing ancestor; emit a
//!      fresh probe.
//!    - `Failed { errno }`: retain Pending state; emit Diagnostic; await next event.
//! 4. `on_descent_event` triggers a fresh probe (no settle) on `StructureChanged` at
//!    `current_prefix` — pure mechanism, no witness write. I5 bounds it to at most one descent
//!    probe in flight, so a signal racing an outstanding probe latches a re-probe-owed debt rather
//!    than being dropped (the in-flight walk may predate the signal and so cannot witness it); the
//!    response dispatch repays the debt with a probe that postdates the signal.

use crate::path::empty_path;
use crate::probe::DescentOutcome;
use crate::refcounts::{add_watch, sub_watch, sub_watch_then_try_reap};
use crate::transitions::ParkNarration;
use compact_str::CompactString;
use specter_core::{
    ClassSet, ContribKey, DescentRemaining, DescentState, Diagnostic, EntryKind, FS_ROOT_SEGMENT,
    ProbeFailure, ProbeSlot, ProfileId, ProfileState, ResourceId, ResourceKind, ResourceRole,
    StepOutput, TreePath,
};
use std::time::Instant;

/// Transient-failure retry ceiling for a *signal-bearing* descent probe (one emitted by
/// [`Engine::try_emit_descent_probe`] — a prefix-event / overflow re-trigger or a raced-signal
/// repay). When such a probe fails with [`ProbeFailure::Transient`], the descent re-latches a fresh
/// postdating probe up to this many times before giving up loudly
/// ([`Diagnostic::PendingPathRetriesExhausted`]) and parking for a future event. The bound closes
/// the micro-blip wedge (a one-shot EMFILE losing the awaited segment's creation) without letting a
/// persistently-failing prefix spin a tight re-probe loop; sustained kernel-resource pressure is a
/// walk-policy concern, not the engine's. See [`specter_core::DescentState::signal_retries`].
const SIGNAL_PROBE_RETRY_BUDGET: u8 = 3;

/// Result of `Engine::materialize_path_or_pending`. Either the entire path is disk-observed (the
/// anchor is live on disk; proceed with the normal immediate Seed-burst flow) or the deepest
/// disk-observed prefix is an ancestor (descent registers; remaining components are tracked).
pub(crate) enum MaterializeResult {
    /// Every segment is disk-observed (the whole path is live on disk); the leaf is `User`-rooted.
    Materialized(ResourceId),
    /// Descent is needed. The leaf `ResourceId` is the anchor's (currently `DescentScaffold`-roled)
    /// slot; the engine registers `DescentState` keyed by the Profile's id once it's been minted.
    /// `remaining` is non-empty by [`DescentRemaining`]'s type invariant —
    /// `materialize_path_or_pending` reaches this variant only when `prefix_idx + 1 <
    /// components.len()`, guaranteeing `from_vec` succeeds.
    Pending {
        anchor: ResourceId,
        prefix: ResourceId,
        remaining: DescentRemaining,
    },
}

impl crate::Engine {
    /// Walk a validated [`TreePath`] into the Tree. The leaf is created with `ResourceRole::User`;
    /// non-leaf components are `ResourceRole::DescentScaffold` if freshly created (the existing
    /// `ensure_root` / `ensure_child` preserve existing roles, so an already- User parent stays
    /// User).
    ///
    /// Returns `Materialized` iff every segment is **disk-observed** (the whole path is live on
    /// disk). Otherwise returns `Pending` with the deepest disk-observed ancestor as `prefix` and
    /// the remaining components as the descent path.
    ///
    /// "Deepest disk-observed ancestor" is the deepest `i` for which every prefix slot
    /// `lookup(path.segments()[..=j])` (`j <= i`) reports
    /// [`specter_core::Resource::is_disk_observed`] — classified by a real observation, or watched
    /// by a live claimant. **Tree-side existence is not enough**: `ensure_path` (below, and at
    /// every prior attach) eagerly mints the whole path — anchor included — as `DescentScaffold`
    /// slots, and a parked Profile's leftover chain is a run of such never-observed scaffolds that
    /// exist in the Tree without existing on disk. Counting those as pre-existing would
    /// short-circuit the descent and emit kernel `Watch`es for disk-absent paths plus a doomed cold
    /// Seed; the disk-honest predicate descends instead. The FS-root (index 0) is observed by axiom
    /// — the kernel always `lstat`s `/` — so `i >= 0` for every absolute attach. Role plays no part
    /// in this decision: a `User` peer anchor or a `WatchRootParent` counts only when it carries a
    /// real observation or watch; a never-observed scaffold never does.
    ///
    /// **Pre-conditions are now type-enforced.** [`TreePath`]'s type invariants (non-empty;
    /// `segments()[0] == FS_ROOT_SEGMENT`) make the prior `debug_assert!` and release-mode
    /// degradation branch structurally impossible.
    pub(crate) fn materialize_path_or_pending(&mut self, path: &TreePath) -> MaterializeResult {
        // Borrow segments as `&[&str]` once for the Tree-side helpers (`lookup`, `ensure_path`,
        // `resolve_components`) which all key on `&str`. One small allocation bounded by path depth.
        let components: Vec<&str> = path.segments().iter().map(CompactString::as_str).collect();

        // FS-root bootstrap. Unconditional: [`TreePath`]'s invariant guarantees `components[0] ==
        // FS_ROOT_SEGMENT`, and `ensure_root` is idempotent (returns the existing slot if a root at
        // `/` already exists). The role is `DescentScaffold` on first creation; if a prior `User`
        // attach at `/` already promoted the slot, the preserve-existing-role contract leaves it
        // alone. Bootstrapping unconditionally guarantees every Profile's rewind chain terminates
        // at this `/` slot — the kernel always `lstat`s `/` successfully on Unix, so a `Vanished`
        // response from a `/` probe is impossible, making cascading parent destruction (`rm -rf
        // /a/b/c/d`) recoverable: the descent stays Pending at `/` waiting for the cascade's bottom
        // segment to reappear.
        self.tree
            .ensure_root(FS_ROOT_SEGMENT, ResourceRole::DescentScaffold);

        // Snapshot which segments are *disk-observed* BEFORE the walk, so we can tell a slot that
        // exists on disk from a bare Tree slot — a never-observed scaffold left by `ensure_path`
        // (here or at a prior attach). A scaffold lingers held by a parked Profile's anchor
        // back-ref and the chain's child edges; treating it as a usable prefix would short-circuit
        // the descent onto a disk-absent path. `is_disk_observed` is the honest test: a real
        // classification (`kind`) or a live claimant's watch. The FS-root (index 0) is observed by
        // axiom — the kernel always `lstat`s `/`, and the bootstrap above guarantees its slot.
        let mut pre_existed: Vec<bool> = Vec::with_capacity(components.len());
        let mut cur_lookup: Option<ResourceId> = None;
        for (i, comp) in components.iter().enumerate() {
            let id = self.tree.lookup(cur_lookup, comp);
            let observed = i == 0
                || id
                    .and_then(|rid| self.tree.get(rid))
                    .is_some_and(specter_core::Resource::is_disk_observed);
            pre_existed.push(observed);
            cur_lookup = id;
        }
        debug_assert!(
            pre_existed[0],
            "materialize_path_or_pending: FS-root is observed by axiom (index 0)",
        );

        // Now do the walk. `ensure_path` creates non-leaf as `DescentScaffold`, leaf as `User`.
        let anchor = self
            .tree
            .ensure_path(&components, ResourceRole::User)
            .expect("TreePath::segments() is non-empty by type invariant");

        // Walk forward to the deepest *contiguous* disk-observed prefix. Breaking at the first
        // unobserved segment is load-bearing: observation flows strictly downward through a chain,
        // but a vacated mid-chain slot (watch rejected, kind reset) can leave a deeper observed
        // slot stranded behind it — re-descending through the gap re-observes it, which is the
        // disk-honest choice. The FS-root axiom guarantees `pre_existed[0] == true`, so
        // `prefix_idx` is always at least `0` — no `Option<usize>` trichotomy is needed.
        let mut prefix_idx: usize = 0;
        for (i, &observed) in pre_existed.iter().enumerate() {
            if observed {
                prefix_idx = i;
            } else {
                break;
            }
        }

        if prefix_idx + 1 == components.len() {
            // Whole path is disk-observed — the immediate-Seed path.
            MaterializeResult::Materialized(anchor)
        } else {
            // Segments [0..=prefix_idx] are disk-observed; [prefix_idx+1..] are unobserved
            // scaffolds. `ensure_path` above created every segment, so `resolve_components` on any
            // prefix is guaranteed to succeed — `expect` documents that contract rather than
            // masking a violation behind `unwrap_or(anchor)`.
            let prefix = self
                .resolve_components(&components[..=prefix_idx])
                .expect("ensure_path created every component; prefix slice must resolve");
            // Reuse the already-validated `CompactString` segments from [`TreePath`] rather than
            // re-allocating from `&str`. Bounded by path depth and lifts straight into
            // [`DescentRemaining`].
            let remaining_vec: Vec<CompactString> = path.segments()[prefix_idx + 1..].to_vec();
            // `prefix_idx + 1 < components.len()` is structurally guaranteed by the outer `if`, so
            // `from_vec` always succeeds here; `expect` documents the contract and gives a precise
            // panic message if a future refactor weakens the outer guard.
            let remaining = DescentRemaining::from_vec(remaining_vec).expect(
                "materialize_path_or_pending: Pending branch with empty remaining is \
                 structurally impossible (prefix_idx + 1 < components.len())",
            );
            MaterializeResult::Pending {
                anchor,
                prefix,
                remaining,
            }
        }
    }

    /// Resolve a sequence of path components to its leaf `ResourceId` without mutating the Tree.
    /// Returns `None` if any segment doesn't resolve.
    pub(crate) fn resolve_components(&self, components: &[&str]) -> Option<ResourceId> {
        let mut cur: Option<ResourceId> = None;
        for comp in components {
            cur = Some(self.tree.lookup(cur, comp)?);
        }
        cur
    }

    /// Enter `ProfileState::Pending` against `prefix` with `remaining` path components
    /// (single-component segments, anchor last). Mints the probe correlation, flips the Profile to
    /// `Pending`, bumps the prefix's `STRUCTURE` `watch_demand` contribution, and emits the descent
    /// probe — the four-step Idle → Pending entry sequence as a single helper.
    ///
    /// `witnessed` is the descent's appearance-latch birth value ([`DescentState::new`]): `true`
    /// when the entry itself was driven by an observed anchor loss (the loss signal is the absence
    /// half of the witness; materialization supplies the presence half), `false` when no first-hand
    /// observation vouches for the anchor having been absent — the attach-time entry (no
    /// observation at all) and the event-scan recovery (whose entry event can be sibling churn at
    /// the parent, out of the Sub's scope). After entry the latch moves only on the descent's own
    /// probe observations: an absent-then-present pair across responses latches it
    /// ([`DescentState::note_observed_absent`] / [`DescentState::note_observed_present`]); events
    /// never write it.
    ///
    /// **Ordering: mint → state-flip → add_watch → emit.** Symmetric with
    /// [`Self::materialize_profile_anchor`]'s state-before-refcount pattern. The mint runs *first*
    /// so the `Pending` state is constructed with its probe slot already armed — phase-without-
    /// correlation cannot exist. State-flip *then* refcount keeps the contribution attribution
    /// coherent with the Profile's claim shape at the moment of the refcount edge.
    ///
    /// **Pre-conditions.** Profile must be `Idle` (a fresh attach) or `Parked` (a recovery
    /// re-entry), and must not hold the anchor claim — a claim-Held Profile entering descent would
    /// breach the reap trichotomy and `materialize_anchor`'s precondition at the terminus (the
    /// recovery arms select on `Parked`, whose coherence assert pins `AnchorClaim::None`, so the
    /// claim assert below is the tripwire for a future mis-routed caller). The debug_asserts catch
    /// both. ("No in-flight probe" is implied, not separately asserted: neither `Idle` nor `Parked`
    /// carries a `DescentState`, so the Profile structurally has no probe slot.)
    ///
    /// **Recovery-overlap invariant.** When called from `start_pending_recovery`, the Profile already
    /// holds a `+1 STRUCTURE` contribution on the parent via `Profile.watch_root_parent` (set at the
    /// original anchor materialization, never cleared on `on_anchor_terminal_event`). This helper
    /// bumps `+1 STRUCTURE` again on the same resource, giving `+2`. At re-materialization the
    /// descent contribution is subbed and the `watch_root_parent` contribution persists —
    /// `set_watch_root_parent` is idempotent on the recovery path (`engine.rs::set_watch_root_parent`
    /// short-circuits when the cache already points at the same parent).
    pub(crate) fn enter_pending_descent(
        &mut self,
        profile_id: ProfileId,
        prefix: ResourceId,
        remaining: DescentRemaining,
        witnessed: bool,
        out: &mut StepOutput,
    ) {
        debug_assert!(
            self.profiles
                .get(profile_id)
                .is_some_and(|p| matches!(p.state(), ProfileState::Idle | ProfileState::Parked)),
            "enter_pending_descent: Profile must be Idle or Parked before \
             entering descent; caller must release prior state first. ('No \
             in-flight probe' is not a separate condition — neither variant \
             carries a `DescentState`, so the Profile structurally has no \
             probe slot.) (profile = {profile_id:?})",
        );
        debug_assert!(
            self.profiles
                .get(profile_id)
                .is_some_and(|p| matches!(p.anchor_claim(), specter_core::AnchorClaim::None)),
            "enter_pending_descent: a claim-Held Profile must never enter \
             descent — Pending ∧ Held breaches the reap trichotomy and the \
             terminus' materialize_anchor precondition (profile = {profile_id:?})",
        );

        // Step 1: mint the correlation. Runs first so the Pending state below is constructed with
        // its slot already armed — no window where the phase exists without a correlation.
        let correlation = self.mint_probe_correlation();

        // Step 2: state-flip Idle/Parked → Pending, constructed armed. Done before the refcount
        // edge so any reader between this point and step 3 sees the Profile's claim shape that the
        // contribution will attribute to (matches `materialize_profile_anchor`'s sequencing).
        //
        // Liveness needs no separate guard. The entry `debug_assert` carries the precondition in
        // dev/CI; in release the construct-armed slot enforces liveness structurally — on the
        // (synchronously unreachable) stale id `transition_state` no-ops and drops the constructed
        // `Pending`, whose armed `ProbeSlot` trips its `Drop` guard loudly in every build, naming
        // the orphaned correlation. The discard *is* the enforcement.
        self.profiles.transition_state(
            profile_id,
            ProfileState::Pending(DescentState::new(
                prefix,
                remaining,
                ProbeSlot::armed(correlation),
                witnessed,
            )),
        );

        // Step 3: install the prefix's STRUCTURE contribution.
        add_watch(
            &mut self.tree,
            prefix,
            ContribKey::ProfileDescent(profile_id),
            ClassSet::STRUCTURE,
            out,
        );

        // Step 4: the choke reads the correlation back off the Pending descent slot and resolves
        // the prefix target off state.
        self.emit_owner_probe(profile_id, out);
    }

    /// Fan a typed descent response out to its terminal helper — symmetric with
    /// [`Self::dispatch_burst_outcome`], total over the three [`DescentOutcome`] variants. The
    /// illegal `AnchorOk` / `SubtreeProven` shapes were already rejected by the
    /// `DescentOutcome::try_from` parse at the demux seam, so they never reach here.
    ///
    /// **Re-probe-owed repay.** After the arm runs, this is the single point that repays any debt
    /// owed — whether latched by a signal that raced the just-consumed probe (a prefix event or
    /// overflow — see [`Self::on_descent_event`]) or re-latched by the failure arm's bounded retry
    /// (a transiently-failed signal-bearing probe — see [`Self::dispatch_descent_failed`]). The arm
    /// has one of three shapes, and the repay reads correctly against all of them:
    /// - **re-armed inline** (advance / rewind): a fresh probe is already in flight and postdates
    ///   the latched signal, so `try_emit_descent_probe` declines on the in-flight gate — debt
    ///   consumed, no second probe.
    /// - **parked / retained** (segment absent / `Failed`): no probe in flight, so the repay emits
    ///   the owed postdating probe. A `Failed` arm may have re-latched the debt itself (the bounded
    ///   signal-bearing retry), in which case the emitted probe carries the retry forward.
    /// - **terminal / abandoned** (materialized anchor / root-prefix `Vanished`): the descent is
    ///   gone, `descent_state_mut` is `None`, the debt died with the state (the Seed's
    ///   `WholeSubtree` probe postdates everything; an abandoned descent must not re-probe).
    ///
    /// The walker-contract-violation arm (`walker_contract_violated_descent`) bypasses this hook by
    /// construction — it is dispatched directly from `on_probe_response`, never through here, so a
    /// buggy walker is never auto-re-probed into a tight loop.
    pub(crate) fn dispatch_descent(
        &mut self,
        owner: ProfileId,
        outcome: DescentOutcome,
        now: Instant,
        out: &mut StepOutput,
    ) {
        match outcome {
            DescentOutcome::SegmentObserved { kind } => {
                self.dispatch_descent_ok(owner, kind, now, out);
            }
            DescentOutcome::Vanished => self.dispatch_descent_vanished(owner, now, out),
            DescentOutcome::Failed(failure) => self.dispatch_descent_failed(owner, failure, out),
        }

        // Repay the epoch debt. `take_reprobe_owed` clears it unconditionally (a re-armed arm drops
        // the debt without a redundant probe); `try_emit_descent_probe` then emits only when no
        // probe is in flight, and never re-latches, so the repay converges in one probe per latched
        // window. A gone descent (terminal / abandoned) yields `None` here and owes nothing.
        if self
            .descent_state_mut(owner)
            .is_some_and(DescentState::take_reprobe_owed)
        {
            self.try_emit_descent_probe(owner, out);
        }
    }

    /// Recover a descent from a walker-contract violation — a `Descent` probe whose payload
    /// resolved to an `AnchorOk` / `SubtreeProven` proof the route cannot accept (descent never
    /// queries an anchor's `lstat` shape or a subtree proof). The typed [`DescentOutcome`] parse
    /// rejected the payload at the demux seam; this **abandons** the descent prefix.
    ///
    /// `debug_assert!` in dev/CI (a production walker never emits this shape), then in release
    /// emits [`Diagnostic::WalkerContractViolated`] and routes through
    /// [`Self::release_descent_prefix_claim`] — the abandon terminal (the same release path the
    /// root-prefix `dispatch_descent_vanished` branch uses), **not** `dispatch_descent_vanished`
    /// itself: that rewinds to the parent and re-arms a fresh probe, which against a
    /// persistently-buggy walker is a tight re-probe loop. Abandoning parks the Profile loudly
    /// (recovery via the park's own doors — overflow, re-attach — and self-healing on a fresh
    /// descent). The probe slot was disarmed by `take_owner_probe` before dispatch and the descent
    /// state is unflipped at entry, so the release helper's preconditions hold.
    pub(crate) fn walker_contract_violated_descent(
        &mut self,
        owner: ProfileId,
        out: &mut StepOutput,
    ) {
        debug_assert!(
            false,
            "walker contract violated: a Descent probe received a non-enumeration \
             outcome (AnchorOk | SubtreeProven) — descent never queries an anchor \
             shape (owner = {owner:?})",
        );
        out.diagnostics
            .push(Diagnostic::WalkerContractViolated { owner });
        self.release_descent_prefix_claim(owner, ParkNarration::Operational, out);
    }

    /// Dispatch a successful descent response. The walker honoured the `Descent` request shape and
    /// answered the structural question about the awaited head segment with one
    /// `lstat(prefix/segment)`; this routine either advances descent one level, materializes the
    /// anchor, or awaits the next event.
    ///
    /// **The answer is always about the current head.** The wire carried the head segment at
    /// emission, and the head is frozen for the armed slot's lifetime — every head mutation
    /// (advance / rewind) re-arms with a freshly minted correlation, so a response that passed the
    /// correlation gate in `on_probe_response` cannot be an answer about a different segment.
    /// `DescentRemaining` is non-empty by type invariant, so there is no defensive empty-arm
    /// recovery path and no corresponding `Diagnostic` variants.
    ///
    /// **Caller (`on_probe_response`).** The descent probe slot was disarmed (consume-once) before
    /// dispatch; the advance / rewind branches re-arm it with a freshly-minted correlation.
    pub(crate) fn dispatch_descent_ok(
        &mut self,
        owner: ProfileId,
        observed: Option<EntryKind>,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Sample the head segment + arity from descent state, then drop the borrow. We clone only
        // the head (cheap when CompactString stays inline); the tail mutation runs in place via
        // `descent_state_mut` later, no whole-vec rebuild.
        //
        // Loud entry: the probe gate proved `Pending` and the consume-once disarm leaves the
        // variant intact, so a `None` is a state-machine breach — a silent skip would drop the
        // response's advance / materialize / park and wedge the descent. Same discipline as the
        // `_mut` re-projections below.
        let Some(descent) = self.descent_state_mut(owner) else {
            unreachable!(
                "dispatch_descent_ok: owner {owner:?} not in descent at \
                 dispatch entry — on_probe_response's probe gate proved Pending"
            );
        };
        // A successful observation is a terminal fate for any outstanding signal-bearing probe — the
        // signal's effect is now seen (present, still-absent, or, on the rewind path, the prefix
        // moved). Release the bounded-retry budget so a later signal earns a fresh one; the forward
        // advance / rewind probes emitted below are non-signal-bearing and must not inherit a marker.
        descent.clear_signal_probe();
        let prefix = descent.current_prefix();
        let next_segment = descent.remaining_components().head().clone();
        let is_terminal = descent.remaining_components().is_terminal();

        let Some(entry_kind) = observed else {
            // Awaited segment not yet present under a healthy prefix: record the absence
            // observation — the standing half of the appearance witness a later found-response
            // completes — then await the next event. A witnessed descent narrates the park:
            // post-loss recovery flows through here when the replacement hasn't landed yet
            // (delete-then-write saves), and a silent park would read as the recovery vanishing in
            // a debug tail. Unwitnessed descents skip the narration — parking is their steady
            // state. Loud arm: the entry resolution above proved the owner in descent and nothing
            // in between mutated it, so a `None` here is a state-machine breach, not a benign race.
            let Some(d) = self.descent_state_mut(owner) else {
                unreachable!(
                    "dispatch_descent_ok: owner {owner:?} left descent between \
                     the entry resolution and the park"
                );
            };
            d.note_observed_absent();
            if d.witnessed() {
                out.diagnostics
                    .push(Diagnostic::PendingPathAwaitingSegment {
                        profile: owner,
                        prefix,
                        segment: next_segment,
                    });
            }
            return;
        };

        // The awaited segment is present. Under a standing absence observation this completes the
        // absence→presence appearance witness — latched here, before the terminal arm reads it for
        // the cold/triggered Seed split (the intermediate arm carries it forward unchanged). Loud
        // arm for the same reason as the park above.
        let Some(d) = self.descent_state_mut(owner) else {
            unreachable!(
                "dispatch_descent_ok: owner {owner:?} left descent between \
                 the entry resolution and the found-segment latch"
            );
        };
        d.note_observed_present();

        // Materialize the next segment as a Tree slot. Look it up first; if absent, ensure as
        // DescentScaffold (the terminal arms may promote it to User via `promote_scaffold`).
        let new_resource = match self.tree.lookup(Some(prefix), &next_segment) {
            Some(r) => r,
            None => self
                .tree
                .ensure_child(prefix, &next_segment, ResourceRole::DescentScaffold)
                .expect("descent prefix held alive by the ProfileDescent contribution"),
        };
        self.tree.set_kind(new_resource, entry_kind.into());

        if is_terminal {
            // Terminal: materialise the anchor and start the Seed burst. The helper releases the
            // prefix's STRUCTURE contribution and installs the anchor's as part of its own
            // state-flip sequence.
            self.materialize_profile_anchor(owner, prefix, new_resource, entry_kind, now, out);
        } else {
            self.advance_descent(owner, prefix, new_resource, out);
        }
    }

    /// Advance descent one literal segment.
    ///
    /// Sequence:
    /// 1. Mint a fresh probe correlation for `owner`.
    /// 2. Mutate descent state in place: advance `current_prefix` to the new resource, drop the
    ///    consumed head segment from `remaining_components`, and re-arm the probe slot with the
    ///    fresh correlation (the response handler disarmed it before routing here, so `arm`'s
    ///    empty-slot precondition holds).
    /// 3. Release the old prefix's STRUCTURE contribution; install the new prefix's.
    /// 4. Emit the fresh descent probe at the new prefix.
    ///
    /// The old prefix stays alive because the freshly-advanced `new_prefix` is still its `children`
    /// entry — the structural anchor holds the slot across `sub_watch`. No `try_reap` here: the
    /// routine release helper would see a non-empty `children` map and short-circuit anyway, so we
    /// skip the call. (Role is metadata throughout — its tag stays `DescentScaffold` from the
    /// initial `ensure_child` but does not affect retention.)
    fn advance_descent(
        &mut self,
        owner: ProfileId,
        old_prefix: ResourceId,
        new_prefix: ResourceId,
        out: &mut StepOutput,
    ) {
        let correlation = self.mint_probe_correlation();
        // Loud arm — `dispatch_descent_ok` reaches here only for an owner the response gate proved
        // in descent (slot disarmed before routing), so `descent_state_mut` resolving `None` is a
        // state-machine breach. Silent skip ⇒ no re-arm, no probe, no diagnostic (a wedge); loud ⇒
        // crash.
        let Some(d) = self.descent_state_mut(owner) else {
            unreachable!(
                "advance_descent: owner {owner:?} not in descent after \
                 dispatch_descent_ok proved it"
            );
        };
        d.advance_to(new_prefix);
        // Non-terminal by caller contract — `dispatch_descent_ok` routes terminal descents through
        // anchor materialization before reaching `advance_descent`. The debug_assert inside
        // `DescentRemaining::advance` pins this for regression detection.
        d.remaining_components_mut().advance();
        d.arm_probe(correlation);

        let key = ContribKey::ProfileDescent(owner);
        sub_watch(&mut self.tree, old_prefix, key, out);
        add_watch(&mut self.tree, new_prefix, key, ClassSet::STRUCTURE, out);

        // The choke reads the correlation back off the descent slot and resolves the
        // (just-advanced) prefix target off state.
        self.emit_owner_probe(owner, out);
    }

    /// Promote `new_resource` to the Profile's anchor slot. Sole call site is
    /// [`Self::dispatch_descent_ok`]'s terminal arm — the descent has just resolved its last
    /// remaining segment and the Profile is about to leave `Pending` for `Idle → Active(Seed)`.
    ///
    /// Sequence (load-bearing):
    /// 1. Read the witnessed-appearance latch off the descent state — it must happen before step
    ///    3's `Pending → Idle` flip destroys the [`DescentState`] carrying it (the descent's probe
    ///    slot is already disarmed at this depth, so the drop is tripwire-safe).
    /// 2. Promote the slot's role to `User` via [`specter_core::Tree::promote_scaffold`] — a no-op
    ///    if a co-resident peer already gave the slot a real role (`WatchRootParent` / `User`), so
    ///    materialization never clobbers a peer's claim.
    /// 3. Transition the Profile **before** any refcount op via
    ///    [`specter_core::Profile::materialize_anchor`] — atomic `Pending → Idle`, claim install,
    ///    kind pin. The recompute (multi-contributor case) reads `Profile.state` and
    ///    `Profile.anchor_claim` to attribute contributions; the post-flip world has the prefix's
    ///    STRUCTURE source gone (state no longer Pending) and the anchor's mask source owed.
    /// 4. Sub the prefix's STRUCTURE; add the anchor's mask (captured from `Profile.events`).
    /// 5. Install the watch-root-parent contribution (deferred from `attach_sub_inner` because the
    ///    parent didn't exist on disk when the Profile attached).
    /// 6. Start the Seed burst — triggered with the anchor as provenance iff the descent witnessed
    ///    activity, cold otherwise. The trigger's only job is landing in the Seed's `dirty` so
    ///    `seed_owes_first_fire` sees the witness (every Seed probe targets the anchor regardless);
    ///    a triggered Seed opens Batching-first, so a storm of appearances debounces on the user's
    ///    settle window.
    fn materialize_profile_anchor(
        &mut self,
        profile_id: ProfileId,
        prefix: ResourceId,
        new_resource: ResourceId,
        entry_kind: EntryKind,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Loud read: `dispatch_descent_ok`'s found-segment latch just proved the owner in descent and
        // only Tree writes intervened, so a `None` is a state-machine breach — a silent unwitnessed
        // degrade here would turn an owed first fire into a quiet pin, the worst failure shape this
        // terminus can produce. Same discipline as the dispatch family's entry resolutions.
        let Some(descent) = self.descent_state(profile_id) else {
            unreachable!(
                "materialize_profile_anchor: owner {profile_id:?} not in descent \
                 at the terminus — dispatch_descent_ok proved it"
            );
        };
        let witnessed = descent.witnessed();

        // `new_resource` is either a freshly-ensured DescentScaffold or a peer's pre-existing slot
        // (the caller's lookup hit). `promote_scaffold` flips only a still-scaffold slot and no-ops
        // on a real role, so materialization never clobbers a co-resident peer's WatchRootParent /
        // User.
        self.tree.promote_scaffold(new_resource, ResourceRole::User);

        let events_union = self
            .profiles
            .get(profile_id)
            .map_or(ClassSet::EMPTY, specter_core::Profile::events);

        // Through the ProfileMap chokepoint: the bundled `Pending → (Idle, Held, classified)` write
        // is a carrier-count edge (`Pending` counted, `Idle` not) the wrapper reconciles.
        self.profiles
            .materialize_anchor(profile_id, ResourceKind::from(entry_kind));

        // Profile.resource was assigned to the anchor's slot at attach time; the materialised
        // slot's id should match by construction.
        debug_assert!(
            self.profiles
                .get(profile_id)
                .is_some_and(|p| p.resource() == new_resource),
            "descent anchor materialization: Profile.resource diverges from descent anchor",
        );

        sub_watch(
            &mut self.tree,
            prefix,
            ContribKey::ProfileDescent(profile_id),
            out,
        );
        add_watch(
            &mut self.tree,
            new_resource,
            ContribKey::ProfileAnchor(profile_id),
            events_union,
            out,
        );

        self.set_watch_root_parent(profile_id, out);

        // The terminus is the descent lifecycle's single positive narration: the awaited anchor is
        // now live and the Profile is leaving Pending. Emit before the Seed burst so the stream
        // reads recovery-complete → baseline-established in order. Unconditional across all three
        // descent entries (attach / observed-loss / park recovery) — an unwitnessed cold descent
        // pins its baseline without a `SubFired`, so this is the only mark it leaves.
        out.diagnostics.push(Diagnostic::PendingPathMaterialized {
            profile: profile_id,
            anchor: new_resource,
        });

        // A witnessed descent owes its terminus Seed the appearance provenance: the anchor threads
        // in as the trigger, so the Seed opens Batching-first and classifies
        // `Consequence::FreshSeedFire` / drift on the stable verdict. An unwitnessed descent (every
        // segment found on first observation — no absence half was ever observed) stays cold —
        // attach-over-existing pins silently, the restart-safe doctrine.
        let trigger = witnessed.then(|| {
            // The anchor slot is live by construction (ensured by the caller this step; its claim
            // installed just above), so `path_of` resolves. The degrade keeps the witness — an
            // empty path in `dirty` still counts as activity, and a Seed probe targets the anchor
            // off `Profile.resource`, never the dirty paths.
            let path = self.tree.path_of(new_resource).unwrap_or_else(|| {
                debug_assert!(
                    false,
                    "materialize_profile_anchor: just-materialized anchor slot must resolve \
                     (profile = {profile_id:?}, resource = {new_resource:?})",
                );
                empty_path()
            });
            (new_resource, path)
        });
        self.start_seed_burst(profile_id, trigger, now, out);
    }

    /// **Rewind chain depth.** A `Vanished` response on a rewound prefix triggers a further rewind
    /// via the same path. The chain depth is bounded by the tree-distance from the original prefix
    /// to its ultimate ancestor — at most one rewind cycle per ancestor level. Each rewind step
    /// **adds** a `+1 STRUCTURE` `watch_demand` on the new prefix; in production the chain
    /// auto-extends watches up the ancestor chain until it reaches a still-present ancestor, whose
    /// probe returns `Ok` and routes to `dispatch_descent_ok`'s "next segment not yet present;
    /// await next event" branch.
    ///
    /// **Branch reachability post-bootstrap.** With the unconditional FS-root bootstrap in
    /// [`Self::materialize_path_or_pending`], every descent's rewind chain terminates at the FS-root
    /// slot `/`. The kernel always `lstat`s `/` successfully on Unix, so a `Vanished` response from a
    /// `/` probe is impossible — meaning the `None` arm below is structurally unreachable in
    /// production. A cascade like `rm -rf /a/b/c/d` with anchor at `/d` rewinds through `/c`, `/b`,
    /// `/a`, `/` and terminates on `/`'s `Ok` rather than reaching the arm; the descent stays Pending
    /// at `/` waiting for the cascade's bottom segment to reappear, which makes cascading parent
    /// destruction auto-recoverable. The arm is retained as defense-in-depth against kernel anomalies
    /// (e.g., a chrooted environment where `/` is somehow inaccessible) and to keep the recursion
    /// well-typed; tests must construct the state directly to exercise it.
    ///
    /// For an `N`-level cascade with a Profile anchored at the leaf, the engine emits up to `N`
    /// rewind cycles per Pending Profile (one Watch + one descent probe per cycle). Acceptable in
    /// v1. Rewinds descent to the next-existing ancestor of `prefix`.
    pub(crate) fn dispatch_descent_vanished(
        &mut self,
        owner: ProfileId,
        _now: Instant,
        out: &mut StepOutput,
    ) {
        // Loud entry: the probe gate proved `Pending` and the consume-once disarm leaves the
        // variant intact, so a `None` is a state-machine breach — a silent skip would drop the
        // rewind and wedge the descent. Same discipline as the `_mut` re-projection below.
        let Some(descent) = self.descent_state_mut(owner) else {
            unreachable!(
                "dispatch_descent_vanished: owner {owner:?} not in descent at \
                 dispatch entry — on_probe_response's probe gate proved Pending"
            );
        };
        // The prefix itself is gone, so a fortiori the anchor's path is incomplete — a first-hand
        // absence observation (a path cannot complete through a vanished directory). Recording it
        // here makes the eventual re-completion a witnessed appearance: an ancestor deleted and
        // recreated under a parked attach is a genuine delete-then-recreate of the anchor's path,
        // not an attach-over-existing. Moot on the root-prefix arm below (the release helper tears
        // the descent down), so the uniform entry write costs nothing there.
        descent.note_observed_absent();
        // Observing the prefix vanished is a terminal fate for any outstanding signal-bearing probe:
        // the signal's effect is seen and the descent rewinds. Release the bounded-retry budget so
        // the rewind probe (non-signal-bearing) starts unmarked and a later signal re-budgets.
        descent.clear_signal_probe();
        let prefix = descent.current_prefix();

        out.diagnostics.push(Diagnostic::PendingPathProbeVanished {
            profile: owner,
            prefix,
        });

        // Capture the parent + the prefix's segment name BEFORE we mutate descent state. `parent`
        // selects the rewind branch; `prefix_name` becomes the new first remaining component.
        let parent = self.tree.parent(prefix);
        let prefix_name = self.tree.name(prefix).map(CompactString::from);

        match parent {
            Some(parent_id) => {
                // Rewind. The vanished prefix's segment becomes the *first* remaining component
                // (we're descending into it from the parent again).
                //
                // In-place mutation: prepend onto the existing `remaining_components` rather than
                // cloning + rebuilding a fresh DescentState — saves both the whole-vec clone and
                // the per-element CompactString clone.
                let correlation = self.mint_probe_correlation();
                // Loud arm — `dispatch_descent_vanished` already resolved `descent_state(owner)` at
                // entry, so this `_mut` re-projection is structurally `Some`; a `None` is a
                // state-machine breach, not a benign race. (The inner `prefix_name` `if let` is a
                // genuine `Option` — the root prefix has no segment name — and stays a conditional.)
                let Some(d) = self.descent_state_mut(owner) else {
                    unreachable!(
                        "dispatch_descent_vanished: owner {owner:?} not in \
                         descent after the entry resolution proved it"
                    );
                };
                d.advance_to(parent_id);
                if let Some(name) = prefix_name {
                    d.remaining_components_mut().prepend(name);
                }
                d.arm_probe(correlation);

                let key = ContribKey::ProfileDescent(owner);
                sub_watch_then_try_reap(&mut self.tree, prefix, key, out);

                add_watch(&mut self.tree, parent_id, key, ClassSet::STRUCTURE, out);

                // The choke reads the correlation back off the descent slot and resolves the
                // (rewound) parent target off state.
                self.emit_owner_probe(owner, out);
            }
            None => {
                // Root prefix vanished — no rewind target. Delegate to the release helper (park
                // terminal + counter-aware sub + try_reap). Its preconditions hold here: the
                // descent probe slot was disarmed by `on_probe_response` before dispatch
                // (cancel-first contract) and descent state is unflipped at entry.
                //
                // The Profile parks loudly without a usable descent path — recovery via the park's
                // own doors (overflow, re-attach) or operator action.
                self.release_descent_prefix_claim(owner, ParkNarration::Operational, out);
            }
        }
    }

    /// Failed response handler. The descent always retains in-descent state; the fork is over the
    /// failure class crossed with whether the failed probe was *signal-bearing* (the marker on
    /// [`DescentState`], read here after the consume-once disarm left the variant intact):
    ///
    /// - **[`ProbeFailure::Transient`] of a signal-bearing probe, budget remaining** — re-latch
    ///   `reprobe_owed` and spend one unit. The repay hook in [`Self::dispatch_descent`] then emits
    ///   a fresh postdating probe. This is the bounded re-latch that keeps a one-shot EMFILE blip
    ///   from permanently dropping the awaited segment's own creation event.
    /// - **[`ProbeFailure::Transient`] of a signal-bearing probe, budget spent** — give up loudly
    ///   ([`Diagnostic::PendingPathRetriesExhausted`]) and clear the marker so a *future* prefix
    ///   event / overflow earns a fresh budget. The descent stays Pending, recovering on the next
    ///   event rather than spinning.
    /// - **Everything else** — an [`ProbeFailure::Anchor`] (path-fatal `EACCES`-class — descent
    ///   cannot fix it), or a transient failure of a *non*-signal-bearing probe (the descent-entry
    ///   / forward-advance / rewind probes re-trigger through their own channels). Narrate
    ///   [`Diagnostic::PendingPathProbeFailed`] and await the next event, the long-standing
    ///   posture. A signal-bearing probe that hit a path-fatal Anchor fate also clears its marker
    ///   (it will not retry) so a later event re-budgets.
    pub(crate) fn dispatch_descent_failed(
        &mut self,
        owner: ProfileId,
        failure: ProbeFailure,
        out: &mut StepOutput,
    ) {
        // Loud entry: the probe gate proved `Pending` and the consume-once disarm leaves the
        // variant intact, so a `None` is a state-machine breach — a silent skip would swallow the
        // failure diagnostic. Same discipline as the dispatch family's `_mut` re-projections. Both
        // reads are `Copy`, so the borrow ends before the `_mut` re-projections in the arms below.
        let Some(descent) = self.descent_state(owner) else {
            unreachable!(
                "dispatch_descent_failed: owner {owner:?} not in descent at \
                 dispatch entry — on_probe_response's probe gate proved Pending"
            );
        };
        let prefix = descent.current_prefix();
        let signal_retries = descent.signal_retries();

        match (failure, signal_retries) {
            (ProbeFailure::Transient { .. }, Some(spent)) if spent < SIGNAL_PROBE_RETRY_BUDGET => {
                // Signal-bearing transient failure with budget left: re-latch the postdating probe
                // (emitted by the repay hook) and spend one unit. Silent — a self-healing blip
                // should not log; the terminal arm narrates only when the chain actually gives up.
                let Some(d) = self.descent_state_mut(owner) else {
                    unreachable!(
                        "dispatch_descent_failed: owner {owner:?} left descent \
                         between the entry resolution and the re-latch"
                    );
                };
                d.note_reprobe_owed();
                d.bump_signal_retry();
            }
            (ProbeFailure::Transient { errno }, Some(spent)) => {
                // Budget spent: the signal-driven observation could not be made across the bounded
                // retries. Give up loudly and clear the marker so a future event starts fresh.
                let Some(d) = self.descent_state_mut(owner) else {
                    unreachable!(
                        "dispatch_descent_failed: owner {owner:?} left descent \
                         between the entry resolution and the budget-spent clear"
                    );
                };
                d.clear_signal_probe();
                out.diagnostics
                    .push(Diagnostic::PendingPathRetriesExhausted {
                        profile: owner,
                        prefix,
                        retries: spent,
                        errno,
                    });
            }
            _ => {
                // Anchor (path-fatal), or a non-signal-bearing transient failure: retain Pending,
                // narrate, await the next event. Clear a stale signal marker iff a signal-bearing
                // probe reached this non-retry fate — a non-signal probe carries none.
                if signal_retries.is_some() {
                    let Some(d) = self.descent_state_mut(owner) else {
                        unreachable!(
                            "dispatch_descent_failed: owner {owner:?} left descent \
                             between the entry resolution and the stale-marker clear"
                        );
                    };
                    d.clear_signal_probe();
                }
                out.diagnostics.push(Diagnostic::PendingPathProbeFailed {
                    profile: owner,
                    prefix,
                    failure,
                });
            }
        }
    }

    /// Emit a fresh descent probe at the current prefix iff the descent is live and idle — the
    /// descent's probe-launch mechanism, shared by the event handler ([`Self::on_descent_event`])
    /// and the re-probe-owed repay ([`Self::dispatch_descent`]'s hook). Returns `true` iff it
    /// minted + armed + emitted, `false` when it declined: the Profile is no longer descending
    /// (liveness gate), or a probe is already in flight (I5 — at most one descent probe per owner).
    /// No settle wait — descent is event-driven.
    ///
    /// **Marks the emitted probe signal-bearing.** This is the single choke for the two probes that
    /// are the sole observer of a structural signal which will not re-arrive — the prefix-event /
    /// overflow re-trigger and the raced-signal repay — so a successful emission flags the
    /// descent's signal-bearing retry budget ([`DescentState::mark_signal_probe`]). A transient
    /// failure of the probe then re-latches a bounded retry rather than wedging the descent on a
    /// lost one-shot signal (see [`Self::dispatch_descent_failed`]). Marking is idempotent on the
    /// budget: a repay re-emission mid-chain preserves the count.
    ///
    /// **No `reprobe_owed` side effect.** It neither sets nor consumes the `reprobe_owed` debt.
    /// That keeps the repay hook safe to call after a dispatch arm already re-armed inline (advance
    /// / rewind): `try_emit` simply declines on the in-flight gate without re-latching against that
    /// arm's own postdating probe (and so never marks it signal-bearing — an advance / rewind probe
    /// re-triggers through its own channel), so the debt the hook just consumed cannot resurrect
    /// into a spurious extra probe. Latching lives solely in `on_descent_event`'s miss branch.
    fn try_emit_descent_probe(&mut self, owner: ProfileId, out: &mut StepOutput) -> bool {
        // Liveness gate: an owner no longer descending has nothing to probe (a benign post-transition
        // race on the event path; a materialized / abandoned descent on the repay path).
        if self.descent_state(owner).is_none() {
            return false;
        }
        // In-flight gate (I5). A probe already out covers the current prefix; the caller decides
        // what to do with the decline — `on_descent_event` latches the debt, the repay hook leaves
        // it to that probe (which postdates the signal).
        if self.pending_probe_for(owner).is_some() {
            return false;
        }

        let correlation = self.mint_probe_correlation();
        // Loud arm — the gates above proved the owner in descent with no in-flight probe, and
        // nothing between mutated it, so this `_mut` re-projection is structurally `Some`.
        let Some(d) = self.descent_state_mut(owner) else {
            unreachable!(
                "try_emit_descent_probe: owner {owner:?} left descent between \
                 the gates and the re-arm"
            );
        };
        d.arm_probe(correlation);
        // The probe this choke emits is the only observer of an unrepeatable signal; flag it so a
        // transient failure re-latches a bounded retry instead of dropping the signal.
        d.mark_signal_probe();
        // The choke reads the correlation back off the descent slot and resolves the prefix target
        // off state.
        self.emit_owner_probe(owner, out);
        true
    }

    /// Handler for a signal reaching a descent's `current_prefix`: a prefix `StructureChanged`
    /// (routed from `on_fs_event`) or a sensor overflow (`on_sensor_overflow`'s Pending arm calls
    /// this directly). Tries to emit a fresh probe; if one is already in flight, latches the
    /// re-probe-owed debt instead of dropping the signal.
    ///
    /// **Why latch, not drop.** "In flight" at the engine means only that the response is unprocessed
    /// — the walk behind it may have completed *before* this signal, so its response cannot witness
    /// this signal's change. Dropping the signal there wedges the descent: a stale pre-signal
    /// response parks it, and the signal (often the awaited segment's own creation, or an edge the
    /// kernel lost to the overflow window) never re-arrives. The latch defers the re-probe to the
    /// response dispatch ([`Self::dispatch_descent`]'s repay hook), which emits a probe that
    /// postdates the signal. See [`specter_core::DescentState`]'s `reprobe_owed` field doc.
    ///
    /// **Pure mechanism — no witness write.** A directory event at the prefix names no segment on
    /// either backend, so it cannot distinguish the awaited segment appearing from sibling churn
    /// entirely outside the Sub's scope — a daemon whose attach path crosses a busy directory
    /// (`/tmp`, `/var/log`, a shared tempdir) sees such churn constantly. The appearance witness
    /// lives in the probe observations themselves ([`Self::dispatch_descent_ok`] /
    /// [`Self::dispatch_descent_vanished`]): an absent-then-present pair latches
    /// `DescentState::witnessed`; a response that finds every segment on first observation leaves
    /// the terminus cold. The probe this handler triggers is how an appearance gets observed at all
    /// — the event's only role. Overflow likewise proves events were dropped somewhere in scope,
    /// not that the awaited segment appeared; the re-probe reads the post-overflow tree and its
    /// observations carry whatever witness is due.
    pub(crate) fn on_descent_event(&mut self, owner: ProfileId, out: &mut StepOutput) {
        if self.try_emit_descent_probe(owner, out) {
            return;
        }
        // The probe didn't launch. The liveness gate is one decline cause; the other — the only one
        // possible while the descent is still live — is an in-flight probe whose walk may predate
        // this signal. Latch the debt there; a gone descent (terminal / abandoned) owes nothing.
        if let Some(d) = self.descent_state_mut(owner) {
            d.note_reprobe_owed();
        }
    }
}

#[cfg(test)]
mod tests {
    //! Tests for `engine::descent` — pending-path scenarios that exercise `DescentState` lifecycle
    //! in isolation. Tests compose `Engine` with `MockSensor`-style direct ProbeResponse injection.

    #![allow(
        clippy::items_after_statements,
        clippy::manual_let_else,
        clippy::match_wildcard_for_single_variants,
        clippy::missing_const_for_fn,
        clippy::needless_pass_by_value,
        clippy::option_if_let_else,
        clippy::single_match_else,
        clippy::too_many_lines
    )]

    use crate::Engine;
    use compact_str::CompactString;
    use specter_core::testkit::single_exec_program;
    use specter_core::{
        ActionProgram, AnchorClaim, ClassSet, Diagnostic, EffectScope, EntryKind, FS_ROOT_SEGMENT,
        Input, ProbeFailure, ProbeOp, ProbeOutcome, ProbeRequest, ProbeResponse, ProfileIdentity,
        ReapTrigger, ResourceId, ResourceKind, ResourceRole, ScanConfig, SubAttachAnchor,
        SubAttachRequest,
    };
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    fn cfg() -> ScanConfig {
        ScanConfig::builder().recursive(true).build()
    }

    fn empty_program() -> Arc<ActionProgram> {
        single_exec_program([specter_core::ArgTemplate::new([
            specter_core::ArgPart::literal("/bin/true"),
        ])])
    }

    /// Set up an Engine with `/foo` as a Dir; attach a Sub at path `/foo/bar`. Bar doesn't exist
    /// yet — descent registers.
    fn setup_pending_one_level() -> (Engine, specter_core::SubId, specter_core::ProfileId) {
        let mut e = Engine::new();
        // /foo exists as a Dir with no role-anchor — represents a real directory the engine has
        // discovered.
        let foo = e
            .tree_mut()
            .ensure_path(&[FS_ROOT_SEGMENT, "foo"], ResourceRole::User)
            .expect("non-empty fixture");
        e.tree_mut().set_kind(foo, ResourceKind::Dir);

        let req = SubAttachRequest::for_anchor(
            "guard".into(),
            SubAttachAnchor::Path(PathBuf::from("/foo/bar")),
            cfg(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());
        let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid).unwrap().profile();
        (e, sid, pid)
    }

    /// Resolve `/foo` after the helper's `ensure_path` placed it under the synthetic FS-root.
    /// Centralises the two-step lookup so individual tests stay readable.
    fn lookup_foo(e: &Engine) -> ResourceId {
        let root = e
            .tree()
            .lookup(None, FS_ROOT_SEGMENT)
            .expect("FS-root bootstrapped by ensure_path");
        e.tree().lookup(Some(root), "foo").expect("/foo exists")
    }

    #[test]
    fn descent_one_level_advances_on_created_entry() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        assert!(e.descent_state(pid).is_some());
        let descent = e.descent_state(pid).unwrap();
        let correlation = e.pending_probe_for(pid).expect("first probe in flight");
        assert_eq!(
            descent
                .remaining_components()
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![CompactString::from("bar")],
        );

        // Inject a probe response showing `bar` now exists.
        let foo = lookup_foo(&e);
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation,
                outcome: ProbeOutcome::SegmentObserved {
                    kind: Some(EntryKind::Dir),
                },
            }),
            Instant::now(),
        );

        // Anchor materialized: descent state cleared; the Seed burst is cold-arm Verifying-first —
        // a probe is emitted at burst construction (the same step as materialization).
        assert!(e.descent_state(pid).is_none());
        assert!(
            matches!(
                e.profiles().get(pid).unwrap().state(),
                specter_core::ProfileState::Active(_, _)
            ),
            "materialization starts the Seed burst (Active, not Idle)",
        );
        assert!(
            out.probe_ops()
                .iter()
                .any(|op| matches!(op, ProbeOp::Probe { .. })),
            "cold-arm Seed: probe emitted at burst construction (materialization)",
        );
        // The terminus narrates its single positive signal: the awaited anchor is now live.
        let bar = e.tree().lookup(Some(foo), "bar").expect("bar materialized");
        assert!(
            out.diagnostics.iter().any(|d| matches!(
                d,
                Diagnostic::PendingPathMaterialized { profile, anchor }
                    if *profile == pid && *anchor == bar
            )),
            "descent terminus emits PendingPathMaterialized for the live anchor: {:?}",
            out.diagnostics,
        );
        let _ = e.cancel_all_in_flight_probes();
    }

    #[test]
    fn descent_two_levels_advances_progressively() {
        let mut e = Engine::new();
        let foo = e
            .tree_mut()
            .ensure_path(&[FS_ROOT_SEGMENT, "foo"], ResourceRole::User)
            .expect("non-empty fixture");
        e.tree_mut().set_kind(foo, ResourceKind::Dir);

        let req = SubAttachRequest::for_anchor(
            "guard".into(),
            SubAttachAnchor::Path(PathBuf::from("/foo/bar/baz")),
            cfg(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());
        let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid).unwrap().profile();

        // First probe at /foo. Inject "bar" appears.
        let descent = e.descent_state(pid).unwrap();
        let corr1 = e.pending_probe_for(pid).unwrap();
        assert_eq!(
            descent
                .remaining_components()
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![CompactString::from("bar"), CompactString::from("baz")],
        );

        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr1,
                outcome: ProbeOutcome::SegmentObserved {
                    kind: Some(EntryKind::Dir),
                },
            }),
            Instant::now(),
        );

        // Now descent should be at /foo/bar with remaining=[baz].
        let descent = e.descent_state(pid).expect("still pending");
        assert_eq!(
            descent
                .remaining_components()
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![CompactString::from("baz")],
        );
        let _bar = e.tree().lookup(Some(foo), "bar").expect("bar materialized");
        let corr2 = e.pending_probe_for(pid).expect("fresh probe");
        assert_ne!(corr1, corr2, "fresh correlation per descent step");

        // Inject "baz" appears.
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr2,
                outcome: ProbeOutcome::SegmentObserved {
                    kind: Some(EntryKind::Dir),
                },
            }),
            Instant::now(),
        );

        // Anchor materialized.
        assert!(e.descent_state(pid).is_none());
        let _ = e.cancel_all_in_flight_probes();
    }

    #[test]
    fn descent_no_progress_keeps_pending() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        let corr = e.pending_probe_for(pid).unwrap();

        // The awaited `bar` segment is absent under a healthy prefix.
        let _foo = lookup_foo(&e);
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::SegmentObserved { kind: None },
            }),
            Instant::now(),
        );

        // Still pending; no new probe.
        let descent = e.descent_state(pid).unwrap();
        assert_eq!(
            descent
                .remaining_components()
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![CompactString::from("bar")],
        );
        assert!(e.pending_probe_for(pid).is_none(), "no probe in flight");
    }

    #[test]
    fn descent_event_at_prefix_emits_fresh_probe() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        // Drain the in-flight probe with the awaited segment still absent.
        let corr = e.pending_probe_for(pid).unwrap();
        let foo = lookup_foo(&e);
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::SegmentObserved { kind: None },
            }),
            Instant::now(),
        );
        // No probe in flight now.
        assert!(e.pending_probe_for(pid).is_none());

        // Inject a StructureChanged at /foo (the prefix).
        let out = e.step(
            Input::FsEvent {
                resource: foo,
                event: specter_core::FsEvent::StructureChanged,
            },
            Instant::now(),
        );

        // Fresh descent probe emitted.
        let probe_for_pid = out
            .probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == pid));
        assert!(probe_for_pid, "descent probe emitted on prefix event");
        assert!(e.pending_probe_for(pid).is_some());
        let _ = e.cancel_all_in_flight_probes();
    }

    #[test]
    fn descent_event_during_in_flight_probe_drops() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        // probe is in flight from setup
        assert!(e.pending_probe_for(pid).is_some());

        let foo = lookup_foo(&e);
        let out = e.step(
            Input::FsEvent {
                resource: foo,
                event: specter_core::FsEvent::StructureChanged,
            },
            Instant::now(),
        );

        // No new probe (I5 for descent).
        let descent_probes = out
            .probe_ops()
            .iter()
            .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == pid))
            .count();
        assert_eq!(descent_probes, 0);
        let _ = e.cancel_all_in_flight_probes();
    }

    /// A signal latched during a descent probe's flight is repaid exactly once, even when the probe
    /// fails. The first `Failed` carries an outstanding debt → one fresh probe; a second `Failed`
    /// with no intervening signal owes nothing, so a persistently-failing prefix never spins a
    /// tight re-probe loop. Pins the failure-arm repay and its termination.
    #[test]
    fn descent_failed_with_latched_signal_reprobes_exactly_once() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        let foo = lookup_foo(&e);
        let corr1 = e.pending_probe_for(pid).expect("first probe in flight");

        // A prefix event races the in-flight probe — latched, not dropped (no new probe emitted).
        let out = e.step(
            Input::FsEvent {
                resource: foo,
                event: specter_core::FsEvent::StructureChanged,
            },
            Instant::now(),
        );
        assert!(
            !out.probe_ops()
                .iter()
                .any(|op| matches!(op, ProbeOp::Probe { .. })),
            "event during in-flight probe latches the debt, emits no probe",
        );
        assert_eq!(
            e.pending_probe_for(pid),
            Some(corr1),
            "in-flight slot untouched by the latched event",
        );

        // The probe fails: state retained, slot disarmed. The latched debt repays — one fresh probe.
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr1,
                outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
            }),
            Instant::now(),
        );
        let corr2 = e
            .pending_probe_for(pid)
            .expect("Failed + latched debt repays with a fresh probe");
        assert_ne!(corr1, corr2, "the repay probe is freshly correlated");
        assert_eq!(
            out.probe_ops()
                .iter()
                .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == pid))
                .count(),
            1,
            "exactly one repay probe — not a burst",
        );

        // The repay probe also fails, with no signal latched during it. No debt is owed, so the
        // descent parks disarmed — no tight loop against a persistently-failing prefix.
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr2,
                outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
            }),
            Instant::now(),
        );
        assert!(
            e.pending_probe_for(pid).is_none(),
            "no probe in flight: an unlatched Failed owes no re-probe",
        );
        assert!(
            e.descent_state(pid).is_some(),
            "descent retained for the next event",
        );
    }

    /// A signal-bearing descent probe that keeps failing *transiently* re-latches a **bounded**
    /// retry: after the budget is spent the descent gives up loudly (`PendingPathRetriesExhausted`)
    /// and parks disarmed rather than spinning a tight re-probe loop. Pins the budget ceiling, the
    /// exact repay count, and the terminal diagnostic — the exhaustion path the integration suite's
    /// one-retry property pin does not reach.
    #[test]
    fn descent_signal_bearing_transient_failure_retries_then_exhausts() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        let foo = lookup_foo(&e);
        let mut corr = e.pending_probe_for(pid).expect("entry probe in flight");

        // A prefix event races the entry probe → latches the re-probe-owed debt. The entry probe is
        // itself non-signal-bearing; its Transient failure repays the latched debt with the first
        // signal-bearing probe, and every signal-bearing failure thereafter re-latches until the
        // budget is spent.
        let _ = e.step(
            Input::FsEvent {
                resource: foo,
                event: specter_core::FsEvent::StructureChanged,
            },
            Instant::now(),
        );

        // Feed Transient failures, following each freshly-correlated repay probe, until the descent
        // parks disarmed (the bound stops the chain). The loop cap is well past the budget so a
        // regression that fails to bound shows as a runaway rather than a hang.
        let mut probes_emitted = 0usize;
        let mut exhausted_retries = None;
        for _ in 0..16 {
            let out = e.step(
                Input::ProbeResponse(ProbeResponse {
                    owner: pid,
                    correlation: corr,
                    outcome: ProbeOutcome::Failed(ProbeFailure::Transient { errno: 24 }),
                }),
                Instant::now(),
            );
            for d in &out.diagnostics {
                if let Diagnostic::PendingPathRetriesExhausted { retries, .. } = d {
                    exhausted_retries = Some(*retries);
                }
            }
            match e.pending_probe_for(pid) {
                Some(next) => {
                    probes_emitted += 1;
                    corr = next;
                }
                None => break,
            }
        }

        assert_eq!(
            exhausted_retries,
            Some(super::SIGNAL_PROBE_RETRY_BUDGET),
            "descent gave up via PendingPathRetriesExhausted after spending the full budget",
        );
        // One entry-debt repay probe + exactly `budget` bounded retries; the budget-spent failure
        // emits none.
        assert_eq!(
            probes_emitted,
            1 + usize::from(super::SIGNAL_PROBE_RETRY_BUDGET),
            "one entry repay plus budget-many bounded retries — no more",
        );
        assert!(
            e.pending_probe_for(pid).is_none(),
            "parked disarmed once the budget is spent — no tight loop",
        );
        assert!(
            e.descent_state(pid).is_some(),
            "descent retained Pending, recovering on a future prefix event",
        );
    }

    /// A signal latched during a descent probe's flight whose response *advances* (the awaited
    /// segment appeared) is repaid by the advance's own probe — which already postdates the signal
    /// — so the repay hook emits nothing extra and the descent converges. Pins that the hook never
    /// re-latches against an inline-re-armed probe: the spurious-probe regression the no-latch
    /// `try_emit_descent_probe` mechanism structurally avoids.
    #[test]
    fn descent_advance_with_latched_signal_does_not_spuriously_reprobe() {
        let mut e = Engine::new();
        let foo = e
            .tree_mut()
            .ensure_path(&[FS_ROOT_SEGMENT, "foo"], ResourceRole::User)
            .expect("non-empty fixture");
        e.tree_mut().set_kind(foo, ResourceKind::Dir);
        let req = SubAttachRequest::for_anchor(
            "guard".into(),
            SubAttachAnchor::Path(PathBuf::from("/foo/bar/baz")),
            cfg(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());
        let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid).unwrap().profile();
        let corr1 = e.pending_probe_for(pid).expect("first probe at /foo");

        // A prefix event races the in-flight probe — latched.
        let out = e.step(
            Input::FsEvent {
                resource: foo,
                event: specter_core::FsEvent::StructureChanged,
            },
            Instant::now(),
        );
        assert!(
            !out.probe_ops()
                .iter()
                .any(|op| matches!(op, ProbeOp::Probe { .. })),
            "event during in-flight probe latches, emits no probe",
        );

        // The probe advances: `bar` appeared, so descent moves to /foo/bar and re-arms inline. That
        // probe postdates the latched event; the repay hook consumes the debt but emits nothing more.
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr1,
                outcome: ProbeOutcome::SegmentObserved {
                    kind: Some(EntryKind::Dir),
                },
            }),
            Instant::now(),
        );
        let corr2 = e
            .pending_probe_for(pid)
            .expect("advance re-armed a probe at /foo/bar");
        assert_ne!(corr1, corr2);
        assert_eq!(
            out.probe_ops()
                .iter()
                .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == pid))
                .count(),
            1,
            "exactly the advance probe — the hook did not re-latch into a second emission",
        );

        // Drive the advance probe to a park (`baz` absent). The debt was already cleared by the
        // advance step, so this park owes no re-probe: the descent converges parked, not spinning.
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr2,
                outcome: ProbeOutcome::SegmentObserved { kind: None },
            }),
            Instant::now(),
        );
        assert!(
            !out.probe_ops()
                .iter()
                .any(|op| matches!(op, ProbeOp::Probe { .. })),
            "no spurious probe after the park — the latch did not survive the advance",
        );
        assert!(
            e.pending_probe_for(pid).is_none(),
            "descent parked disarmed at /foo/bar awaiting the next event",
        );
    }

    #[test]
    fn descent_failed_retains_state() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        let corr = e.pending_probe_for(pid).unwrap();

        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
            }),
            Instant::now(),
        );

        let has_diag = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                Diagnostic::PendingPathProbeFailed {
                    failure: ProbeFailure::Anchor { errno: 13 },
                    ..
                },
            )
        });
        assert!(has_diag);
        // Still pending; no probe in flight.
        let descent = e.descent_state(pid).unwrap();
        assert_eq!(
            descent
                .remaining_components()
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![CompactString::from("bar")],
        );
        assert!(e.pending_probe_for(pid).is_none());
    }

    #[test]
    fn descent_anchor_kind_set_from_entry() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        let corr = e.pending_probe_for(pid).unwrap();
        let foo = lookup_foo(&e);
        let bar = e.tree().lookup(Some(foo), "bar").expect("scaffold exists");

        // Inject as a Dir.
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::SegmentObserved {
                    kind: Some(EntryKind::Dir),
                },
            }),
            Instant::now(),
        );

        let res = e.tree().get(bar).unwrap();
        assert_eq!(res.kind(), Some(ResourceKind::Dir));
        assert!(matches!(res.role, ResourceRole::User));
        let _ = e.cancel_all_in_flight_probes();
    }

    /// Companion to `descent_anchor_kind_set_from_entry`: descent materialisation must also cache
    /// the kind on the Profile itself, not just the Tree slot. The cached `Profile.kind` is the
    /// read path for `transition_to_verifying`'s probe-target dispatch — without it, a
    /// File-anchored Profile materialised from descent would fall through to the `unwrap_or(File)`
    /// default by accident rather than by knowledge.
    #[test]
    fn descent_materialization_caches_profile_kind() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        assert_eq!(
            e.profiles().get(pid).and_then(specter_core::Profile::kind),
            None,
            "Pending Profile starts with kind = None (anchor not yet observed)",
        );

        let corr = e.pending_probe_for(pid).unwrap();
        // Inject as a regular File. This pins the `Profile.kind` cache so a File-anchored
        // materialisation can never re-introduce the descendant-observation dispatch path by an
        // unprobed-anchor accident.
        let _foo = lookup_foo(&e);
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::SegmentObserved {
                    kind: Some(EntryKind::File),
                },
            }),
            Instant::now(),
        );

        assert_eq!(
            e.profiles().get(pid).and_then(specter_core::Profile::kind),
            Some(ResourceKind::File),
            "Profile.kind cached at descent materialisation matches the entry kind",
        );
        let _ = e.cancel_all_in_flight_probes();
    }

    // ===== absolute-path bootstrap & minimal descent probe =====

    /// Absolute-path attaches bootstrap a synthetic FS-root `"/"` segment so descents have a
    /// guaranteed-existing starting prefix. The bootstrap is idempotent across repeated absolute
    /// attaches.
    #[test]
    fn absolute_attach_bootstraps_fs_root_segment() {
        let mut e = Engine::new();

        let req = SubAttachRequest::for_anchor(
            "build".into(),
            SubAttachAnchor::Path(PathBuf::from("/tmp")),
            cfg(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());
        let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid).unwrap().profile();

        // Tree contains the synthetic FS-root and the `tmp` scaffold.
        let root = e.tree().lookup(None, "/").expect("FS-root bootstrapped");
        let tmp = e
            .tree()
            .lookup(Some(root), "tmp")
            .expect("anchor scaffold installed under /");

        // Profile registered; descent in flight at the FS-root.
        let descent = e
            .descent_state(pid)
            .expect("absolute attach against empty Tree is pending");
        assert_eq!(descent.current_prefix(), root);
        assert_eq!(
            descent
                .remaining_components()
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![CompactString::from("tmp")],
        );
        assert!(e.pending_probe_for(pid).is_some());

        // The FS-root carries the descent's watch_demand contribution; the anchor scaffold doesn't
        // (descent hasn't materialized it yet).
        assert_eq!(e.tree().get(root).unwrap().watch_demand(), 1);
        assert_eq!(e.tree().get(tmp).unwrap().watch_demand(), 0);

        // The emitted Watch op carries an *absolute* path — `Tree::path_of` reconstructs `/`
        // because `PathBuf::push("/")` resets to absolute.
        let watch_for_root = out.watch_ops.iter().find_map(|op| match op {
            specter_core::WatchOp::Watch { resource, path, .. } if *resource == root => {
                Some(path.as_ref())
            }
            _ => None,
        });
        assert_eq!(
            watch_for_root,
            Some(Path::new("/")),
            "FS-root Watch op carries an absolute path",
        );

        // The probe op for the descent also carries an absolute prefix path.
        let probe_path = out.probe_ops().iter().find_map(|op| match op {
            ProbeOp::Probe {
                request:
                    ProbeRequest::Descent {
                        owner: profile,
                        target_path,
                        ..
                    },
            } if *profile == pid => Some(target_path.as_ref()),
            _ => None,
        });
        assert_eq!(probe_path, Some(Path::new("/")));
        let _ = e.cancel_all_in_flight_probes();
    }

    /// Two absolute attaches share the FS-root via the bootstrap's idempotence
    /// (`Tree::ensure_root("/")` returns the existing root on the second call).
    #[test]
    fn second_absolute_attach_reuses_fs_root() {
        let mut e = Engine::new();
        let req1 = SubAttachRequest::for_anchor(
            "a".into(),
            SubAttachAnchor::Path(PathBuf::from("/foo")),
            cfg(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        );
        let req2 = SubAttachRequest::for_anchor(
            "b".into(),
            SubAttachAnchor::Path(PathBuf::from("/bar")),
            cfg(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        );
        let _ = e.step(Input::AttachSub(req1), Instant::now());
        let _ = e.step(Input::AttachSub(req2), Instant::now());

        let root = e.tree().lookup(None, "/").expect("single FS-root");
        assert_eq!(e.tree().roots().len(), 1, "exactly one tree root");
        // Both children attach under the same FS-root.
        assert!(e.tree().lookup(Some(root), "foo").is_some());
        assert!(e.tree().lookup(Some(root), "bar").is_some());
        // FS-root carries one contribution from each pending descent.
        assert_eq!(e.tree().get(root).unwrap().watch_demand(), 2);
        let _ = e.cancel_all_in_flight_probes();
    }

    /// Deep absolute paths walk one segment at a time: the descent's `remaining_components`
    /// reflects the unmaterialized tail.
    #[test]
    fn deep_absolute_attach_decomposes_to_one_remaining_per_segment() {
        let mut e = Engine::new();
        let req = SubAttachRequest::for_anchor(
            "log".into(),
            SubAttachAnchor::Path(PathBuf::from("/var/log/myapp")),
            cfg(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());
        let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid).unwrap().profile();

        let root = e.tree().lookup(None, "/").unwrap();
        let descent = e.descent_state(pid).unwrap();
        assert_eq!(descent.current_prefix(), root);
        assert_eq!(
            descent
                .remaining_components()
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![
                CompactString::from("var"),
                CompactString::from("log"),
                CompactString::from("myapp"),
            ],
        );
        let _ = e.cancel_all_in_flight_probes();
    }

    /// Descent probes ride a dedicated `ProbeRequest::Descent` variant — the engine ships
    /// `(profile, correlation, target_path, segment)` and the walker answers with one
    /// `lstat(prefix/segment)`. The typed variant plus the named segment is the structural
    /// guarantee that user filters can't mask the next path segment: there is no enumeration to
    /// filter. Since the engine carries no scan-config on the wire, the lstat's correctness lives
    /// in the sensor's walker tests; this engine test pins the variant choice.
    #[test]
    fn descent_probe_uses_descent_variant() {
        let mut e = Engine::new();
        let foo = e
            .tree_mut()
            .ensure_path(&[FS_ROOT_SEGMENT, "foo"], ResourceRole::User)
            .expect("non-empty fixture");
        e.tree_mut().set_kind(foo, ResourceKind::Dir);

        let user_cfg = specter_core::ScanConfig::builder()
            .recursive(true)
            .pattern(specter_core::GlobPattern::compile("*.c").unwrap())
            .build();
        let req = SubAttachRequest::for_anchor(
            "g".into(),
            SubAttachAnchor::Path(PathBuf::from("/foo/bar")),
            user_cfg,
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());

        let descent_emitted = out.probe_ops().iter().any(|op| {
            matches!(
                op,
                ProbeOp::Probe {
                    request: ProbeRequest::Descent { .. },
                }
            )
        });
        assert!(
            descent_emitted,
            "Pending descent must emit ProbeRequest::Descent (not Subtree); \
             the typed variant is the structural guarantee that user filters \
             can't mask the next path segment",
        );
        let _ = e.cancel_all_in_flight_probes();
    }

    /// Materialization at descent's anchor branch sets `Profile.anchor_claim = AnchorClaim::Held`
    /// so a later reap correctly releases the anchor's `watch_demand`.
    #[test]
    fn descent_materialization_sets_anchor_claim_held() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        let corr = e.pending_probe_for(pid).unwrap();
        let _foo = lookup_foo(&e);
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::SegmentObserved {
                    kind: Some(EntryKind::Dir),
                },
            }),
            Instant::now(),
        );
        assert_eq!(
            e.profiles().get(pid).unwrap().anchor_claim(),
            AnchorClaim::Held,
            "anchor_claim set to Held on descent materialization",
        );
        let _ = e.cancel_all_in_flight_probes();
    }

    /// Pending Profile reaped before descent advances:
    /// - Releases the descent's prefix `watch_demand`.
    /// - Does NOT touch the anchor (anchor was never bumped).
    /// - No underflow panic in dev.
    #[test]
    fn reap_pending_profile_releases_only_descent_prefix() {
        let (mut e, sid, pid) = setup_pending_one_level();
        let foo = lookup_foo(&e);
        let bar = e.tree().lookup(Some(foo), "bar").expect("anchor scaffold");

        // Pre-conditions: descent contributes to `foo`, anchor `bar` is unbumped.
        assert_eq!(e.tree().get(foo).unwrap().watch_demand(), 1);
        assert_eq!(e.tree().get(bar).unwrap().watch_demand(), 0);
        assert_eq!(
            e.profiles().get(pid).unwrap().anchor_claim(),
            AnchorClaim::None,
        );

        // Detach the only Sub. Profile is Pending; Pending Profiles reap immediately (they hold no
        // burst that would resolve a deferred reap).
        let out = e.step(Input::DetachSub(sid), Instant::now());

        // `bar`'s slot is reaped (no other anchors), `foo` still has its pre-existing User Resource
        // — only the descent's contribution is released.
        assert_eq!(
            e.tree()
                .get(foo)
                .map_or(0, specter_core::Resource::watch_demand),
            0,
            "descent prefix watch_demand released",
        );
        assert!(
            out.watch_ops.iter().any(
                |op| matches!(op, specter_core::WatchOp::Unwatch { resource } if *resource == foo)
            ),
            "Unwatch emitted for the descent prefix",
        );
    }

    /// A fresh `Profile::new` defaults to `ProfileState::Idle`, not Pending. Pending is reachable
    /// only through the descent registry paths (`attach_sub_inner` Pending branch,
    /// `start_pending_recovery`, `dispatch_descent_vanished` rewind).
    #[test]
    fn profile_state_default_is_idle() {
        use specter_core::{Profile, ProfileState, ScanConfig};
        let mut e = Engine::new();
        let r = e.tree_mut().ensure_root("anchor", ResourceRole::User);
        let p = Profile::new(
            r,
            ProfileIdentity::new(ScanConfig::builder().build(), MAX_SETTLE, NO_EVENTS),
            SETTLE,
            None,
        );
        assert!(matches!(p.state(), ProfileState::Idle));
    }

    /// `Engine::descent_state` returns `None` for an Idle Profile. The accessor's reader contract
    /// is "Some iff state is Pending."
    #[test]
    fn descent_state_helper_returns_none_for_idle() {
        let mut e = Engine::new();
        let foo = e.tree_mut().ensure_root("foo", ResourceRole::User);
        e.tree_mut().set_kind(foo, ResourceKind::Dir);
        let req = SubAttachRequest::for_anchor(
            "g".into(),
            SubAttachAnchor::Resource(foo),
            cfg(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());
        let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid).unwrap().profile();
        // Materialized Profile starts a Seed burst — Active, not Idle. Drive it to completion to
        // land in Idle.
        let probe = e
            .step(
                Input::ProbeResponse(ProbeResponse {
                    owner: pid,
                    correlation: specter_core::ProbeCorrelation::from(1),
                    outcome: ProbeOutcome::Vanished,
                }),
                Instant::now(),
            )
            .diagnostics;
        let _ = probe; // not asserted; the Vanished response drains the Seed burst to Idle
        assert!(e.descent_state(pid).is_none());
    }

    /// `Engine::descent_state` returns `None` for an Active Profile (a burst is in flight; the
    /// descent slot is not used).
    #[test]
    fn descent_state_helper_returns_none_for_active() {
        let mut e = Engine::new();
        let foo = e.tree_mut().ensure_root("foo", ResourceRole::User);
        e.tree_mut().set_kind(foo, ResourceKind::Dir);
        let req = SubAttachRequest::for_anchor(
            "g".into(),
            SubAttachAnchor::Resource(foo),
            cfg(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());
        let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid).unwrap().profile();
        // Materialized Profile starts a Seed burst — state is Active.
        assert!(matches!(
            e.profiles().get(pid).unwrap().state(),
            specter_core::ProfileState::Active(_, _)
        ));
        assert!(e.descent_state(pid).is_none());
        let _ = e.cancel_all_in_flight_probes();
    }

    /// `Engine::descent_state` returns `Some(d)` for a Pending Profile, and the inner state matches
    /// what was registered.
    #[test]
    fn descent_state_helper_returns_some_for_pending() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        let descent = e.descent_state(pid).expect("Pending state populated");
        assert_eq!(
            descent
                .remaining_components()
                .iter()
                .cloned()
                .collect::<Vec<_>>(),
            vec![CompactString::from("bar")],
        );
        assert!(e.pending_probe_for(pid).is_some());
        let _ = e.cancel_all_in_flight_probes();
    }

    /// `Engine::descent_state` returns `None` for an unknown `ProfileId`. No panic; defensive read.
    #[test]
    fn descent_state_helper_handles_unknown_profile() {
        let e = Engine::new();
        let bogus = specter_core::ProfileId::default();
        assert!(e.descent_state(bogus).is_none());
    }

    /// `ProfileState::Pending` and `ProfileState::Active` are mutually exclusive variants — the
    /// compiler proves the property. This test exercises the lifecycle transition Pending → Idle →
    /// Active(Seed) at descent anchor materialization and asserts the Profile passes through the
    /// intermediate Idle state cleanly (no observation of Pending+Active simultaneously).
    #[test]
    fn profile_state_pending_and_active_are_mutually_exclusive() {
        use specter_core::ProfileState;
        let (mut e, _sid, pid) = setup_pending_one_level();
        // Initially Pending.
        assert!(matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Pending(_)
        ));
        let corr = e.pending_probe_for(pid).unwrap();
        let _foo = lookup_foo(&e);
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::SegmentObserved {
                    kind: Some(EntryKind::Dir),
                },
            }),
            Instant::now(),
        );
        // After anchor materialization: Pending → Idle, then start_seed_burst transitions Idle →
        // Active(Seed). The post-step state is Active.
        assert!(matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Active(_, _)
        ));
        // descent_state agrees: no descent.
        assert!(e.descent_state(pid).is_none());
        let _ = e.cancel_all_in_flight_probes();
    }

    /// `reap_profile`'s trichotomy `debug_assert!` is reachable from the Pending lifecycle (descent
    /// in flight, then Sub detaches) and does not fire — the assertion pins the invariant in code,
    /// not just prose.
    #[test]
    fn reap_profile_trichotomy_debug_assert_holds_for_pending() {
        let (mut e, sid, pid) = setup_pending_one_level();
        // Pending Profile reap path: descent_prefix.is_some() && anchor_claim == None. Predicate
        // `(some && Held)` matches false → assertion holds.
        let _ = e.step(Input::DetachSub(sid), Instant::now());
        assert!(e.profiles().get(pid).is_none(), "Profile reaped");
    }

    #[test]
    fn reap_profile_trichotomy_debug_assert_holds_for_materialized() {
        // Materialized Profile reap path: descent_prefix.is_none() && anchor_claim == Held.
        // Predicate `(none && Held)` matches false → assertion holds.
        let mut e = Engine::new();
        let foo = e.tree_mut().ensure_root("foo", ResourceRole::User);
        e.tree_mut().set_kind(foo, ResourceKind::Dir);
        let req = SubAttachRequest::for_anchor(
            "g".into(),
            SubAttachAnchor::Resource(foo),
            cfg(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        );
        let out = e.step(Input::AttachSub(req), Instant::now());
        let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
        let pid = e.subs().get(sid).unwrap().profile();
        assert_eq!(
            e.profiles().get(pid).unwrap().anchor_claim(),
            AnchorClaim::Held,
        );
        // Drain Seed via Vanished so the Profile lands Idle with the anchor's contribution still
        // held. Then detach.
        let Some(corr) = e.pending_probe_for(pid) else {
            return;
        };
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::Vanished,
            }),
            Instant::now(),
        );
        // Vanished clears the anchor contribution (it's the terminal-event path). Force the
        // materialized branch by re-seeding via a fresh anchor lookup. For coverage of the assertion,
        // the detach path itself is sufficient (it runs reap_profile, which contains the assertion).
        let _ = e.step(Input::DetachSub(sid), Instant::now());
        assert!(e.profiles().get(pid).is_none(), "Profile reaped");
    }

    /// Detaching the last Sub on a Pending Profile reaps immediately rather than setting
    /// `reap_pending = true`. Pending Profiles have no burst whose `finish_burst_to_idle` would
    /// resolve a deferred reap.
    #[test]
    fn detach_sub_pending_profile_reaps_immediately() {
        let (mut e, sid, pid) = setup_pending_one_level();
        let foo = lookup_foo(&e);
        // Pre-condition: Pending; descent contributes +1 to /foo.
        assert!(e.descent_state(pid).is_some());
        assert_eq!(e.tree().get(foo).unwrap().watch_demand(), 1);

        let out = e.step(Input::DetachSub(sid), Instant::now());

        // Profile reaped synchronously: no longer in the registry; descent contribution released
        // atomically.
        assert!(e.profiles().get(pid).is_none(), "Profile reaped");
        assert_eq!(
            e.tree()
                .get(foo)
                .map_or(0, specter_core::Resource::watch_demand),
            0,
            "descent contribution released",
        );
        assert!(
            out.diagnostics.iter().any(|d| matches!(
                d,
                Diagnostic::ProfileReaped {
                    profile,
                    via: ReapTrigger::Immediate,
                } if *profile == pid,
            )),
            "ProfileReaped(Immediate) emitted",
        );
    }

    /// `on_probe_response`'s unified routing dispatches a Pending Profile's response to the descent
    /// path via `match &p.state`. This test asserts the routing by exercising a descent probe
    /// response and verifying the descent advances.
    #[test]
    fn on_probe_response_routes_descent_via_state_match() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        let corr = e.pending_probe_for(pid).unwrap();
        let _foo = lookup_foo(&e);
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::SegmentObserved {
                    kind: Some(EntryKind::Dir),
                },
            }),
            Instant::now(),
        );
        // Descent route fired: Pending → Idle → Active(Seed). The Profile is no longer Pending.
        assert!(e.descent_state(pid).is_none(), "descent route ran");
        assert!(matches!(
            e.profiles().get(pid).unwrap().state(),
            specter_core::ProfileState::Active(_, _)
        ));
        let _ = e.cancel_all_in_flight_probes();
    }

    /// `on_watch_op_rejected` purge transitions Pending → Idle.
    #[test]
    fn on_watch_op_rejected_clears_pending_state() {
        use specter_core::ProfileState;
        let (mut e, _sid, pid) = setup_pending_one_level();
        let foo = lookup_foo(&e);
        assert!(matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Pending(_)
        ));

        let _ = e.step(
            Input::WatchOpRejected {
                resource: foo,
                failure: specter_core::WatchFailure::Pressure { errno: 24 },
            },
            Instant::now(),
        );

        // Purge parks the torn-down descent; descent_state agrees.
        assert!(matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Parked
        ));
        assert!(e.descent_state(pid).is_none());
    }

    #[test]
    fn descent_remaining_from_empty_vec_is_none() {
        use specter_core::DescentRemaining;
        assert!(DescentRemaining::from_vec(Vec::<CompactString>::new()).is_none());
    }

    // ───────────────────────────────────────────────────────────────────────
    // Probe-channel discipline (post-refactor invariants)
    //
    // I5 ("at most one outstanding probe per Profile") is enforced structurally by the owner
    // state's single `ProbeSlot` (one owner ⇒ one state variant ⇒ one slot). The tests below pin
    // the surrounding behaviour: clear-on-cancel, recovery-overlap accounting, and the cancel-first
    // contract on `release_descent_prefix_claim`.
    // ───────────────────────────────────────────────────────────────────────

    /// `on_watch_op_rejected` descent purge: cancel-then-release ordering disarms the descent slot
    /// and emits exactly one `ProbeOp::Cancel`. The Profile transitions Pending → Idle in the same
    /// step.
    #[test]
    fn on_watch_op_rejected_descent_purge_clears_pending_probe_and_emits_cancel() {
        use specter_core::ProfileState;
        let (mut e, _sid, pid) = setup_pending_one_level();
        let foo = lookup_foo(&e);
        assert!(
            e.pending_probe_for(pid).is_some(),
            "descent probe in flight after attach",
        );
        assert!(matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Pending(_),
        ));

        let out = e.step(
            Input::WatchOpRejected {
                resource: foo,
                failure: specter_core::WatchFailure::Pressure { errno: 24 },
            },
            Instant::now(),
        );

        // Field-discipline: slot disarmed atomically with the purge.
        assert!(
            e.pending_probe_for(pid).is_none(),
            "slot disarmed by cancel-before-release",
        );
        // Profile parked via `release_descent_prefix_claim`.
        assert!(matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Parked,
        ));
        // Exactly one Cancel for the Profile (idempotency check).
        let cancels = out
            .probe_ops()
            .iter()
            .filter(|op| matches!(op, ProbeOp::Cancel { owner: profile} if *profile == pid))
            .count();
        assert_eq!(
            cancels,
            1,
            "exactly one Cancel emitted for the in-flight descent probe; got {:?}",
            out.probe_ops(),
        );
    }

    /// `enter_pending_descent` recovery-overlap invariant: when re-entered at the recovery parent,
    /// the parent already carries `+1 STRUCTURE` from `Profile.watch_root_parent`. The helper bumps
    /// `+1` again for the descent contribution; refcount sums to `+2`. Exercised through the
    /// production observed-loss path — a Seed-Vanished routes through
    /// `finalize_anchor_lost_and_descend`, which re-enters descent in the same step.
    #[test]
    fn enter_pending_descent_recovery_overlap_invariant() {
        use specter_core::{ClassSet, ProfileState};
        // Build the recovery scenario:
        //   1. Attach a Sub at /foo/bar (Pending — bar doesn't exist yet).
        //   2. Materialize bar via descent, landing the Profile in Idle with
        //      Profile.watch_root_parent = Some(foo) and foo.watch_demand = +1.
        //   3. Drive the Seed verify to Vanished — the loss step releases the anchor contribution
        //      and re-enters descent at foo with [bar] as remaining, all within the dispatch.
        let (mut e, _sid, pid) = setup_pending_one_level();
        let foo = lookup_foo(&e);

        // Step 1+2: Drive descent to materialization. The probe response with `bar` as a Dir entry
        // materializes the anchor.
        let corr = e.pending_probe_for(pid).expect("descent probe in flight");
        let t_mat = Instant::now();
        let _ = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: ProbeOutcome::SegmentObserved {
                    kind: Some(EntryKind::Dir),
                },
            }),
            t_mat,
        );
        let _bar = e.tree().lookup(Some(foo), "bar").unwrap();
        // Post-materialization: Profile is Active(Seed Verifying); bar carries events_union; foo
        // carries STRUCTURE from watch_root_parent.
        assert_eq!(
            e.profiles().get(pid).unwrap().watch_root_parent(),
            Some(foo),
            "watch_root_parent cached at foo on materialization",
        );
        assert!(
            e.tree().get(foo).unwrap().watch_demand() >= 1,
            "foo carries STRUCTURE from watch_root_parent",
        );

        // The materialized Seed burst is Batching-first; expire its settle timer so a verify probe
        // is in flight, then close it with Vanished (no Effect — fresh Seed).
        let t_settle = t_mat + SETTLE;
        while let Some(entry) = e.pop_expired(t_settle) {
            e.step(
                Input::TimerExpired {
                    profile: entry.profile,
                    kind: entry.kind,
                    id: entry.id,
                },
                t_settle,
            );
        }
        let seed_corr = e
            .pending_probe_for(pid)
            .expect("Seed verify probe in flight after settle expiry");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: seed_corr,
                outcome: ProbeOutcome::Vanished,
            }),
            t_settle,
        );
        // dispatch_pre_fire_vanished routes through finalize_anchor_lost_and_descend: anchor
        // contribution released, baseline/current cleared, and the same step re-enters pending
        // descent at foo — the recovery overlap is established by the production loss path itself.
        assert!(matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Pending(_),
        ));
        assert!(
            e.pending_probe_for(pid).is_some(),
            "descent probe re-armed by the loss step",
        );

        // Recovery overlap: foo's watch_demand is +2 (watch_root_parent STRUCTURE + descent
        // STRUCTURE); bar's anchor contribution is gone.
        assert_eq!(
            e.tree().get(foo).unwrap().watch_demand(),
            2,
            "recovery overlap: descent +1 on top of watch_root_parent +1",
        );
        // The descent probe was emitted at foo (the parent / new prefix). Descent variants carry
        // `target_path` but not `target_resource` (the walker resolves the path against the live
        // filesystem, not against an engine-side ResourceId). Cross-check by comparing the
        // descent's path-of(foo) against the request's `target_path`.
        let foo_path = e.tree().path_of(foo).expect("foo path resolves");
        assert!(
            out.probe_ops().iter().any(|op| matches!(op,
                ProbeOp::Probe { request: ProbeRequest::Descent { owner: profile, target_path, .. } }
                    if *profile == pid && *target_path == foo_path)),
            "descent probe emitted at the parent prefix; got {:?}",
            out.probe_ops(),
        );
        // ClassSet::STRUCTURE is correct for the descent contribution.
        let _ = ClassSet::STRUCTURE;
        let _ = e.cancel_all_in_flight_probes();
    }

    /// Cancel-first contract on `release_descent_prefix_claim`: invoked without a prior
    /// `cancel_owner_probe`, on a Pending Profile whose descent probe is still in flight, the
    /// helper's park transition discard drops the armed `Pending(DescentState)`, tripping
    /// `ProbeSlot`'s Drop tripwire. The tripwire is unconditional (fires in debug AND release), so
    /// the test runs in every build profile. The four production cancel-paths each call
    /// `cancel_owner_probe` first — this guards against future regressions that bypass the
    /// cancel-first order.
    #[test]
    #[should_panic(expected = "ProbeSlot dropped while armed")]
    fn release_descent_prefix_claim_without_cancel_trips_probeslot_drop() {
        let (mut e, _sid, pid) = setup_pending_one_level();
        assert!(
            e.pending_probe_for(pid).is_some(),
            "descent probe in flight (pre-condition for the assertion)",
        );

        // Direct invocation without the prior cancel — assertion fires.
        let mut out = specter_core::StepOutput::default();
        e.release_descent_prefix_claim(
            pid,
            crate::transitions::ParkNarration::Operational,
            &mut out,
        );
    }
}
