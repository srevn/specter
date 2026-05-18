//! Engine diagnostics.
//!
//! Emitted alongside the dropped/clamped Inputs they explain. Variant set
//! grows phase-by-phase as new drop paths land. Each variant is light-weight
//! (a few small fields) and carries enough context to log meaningfully.

use crate::ids::{ProbeCorrelation, ProfileId, PromoterId, ResourceId, SubId, TimerId};
use crate::input::{FsEvent, OverflowScope};
use crate::op::{ProbeOwner, WatchFailure};
use crate::profile::{BurstIntent, ProfileStateDiscriminant};
use crate::resource::ResourceKind;
use compact_str::CompactString;
use std::path::Path;
use std::sync::Arc;

/// Which Profile-side claim was the subject of a [`Diagnostic::ProfileClaimPurged`]
/// emission. Each claim type has a dedicated bookkeeping field on
/// [`crate::profile::Profile`]:
/// - [`Self::Anchor`] ⇔ `Profile.anchor_claim == AnchorClaim::Held`
/// - [`Self::WatchRootParent`] ⇔ `Profile.watch_root_parent == Some(_)`
/// - [`Self::DescentPrefix`] ⇔ `Profile.state == Pending(_)`
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimKind {
    Anchor,
    WatchRootParent,
    DescentPrefix,
}

/// Identifies the burst-lifecycle helper whose precondition failed.
///
/// Tagged onto [`Diagnostic::InvalidBurstTransition`]. Each variant
/// names exactly one helper in `specter-engine`'s `burst.rs`. Variants
/// are added when a helper is created; helpers without a typed
/// precondition (the idempotent `finish_burst_to_idle`) are absent by
/// design.
///
/// The enum is exported from `specter-core` rather than `specter-engine`
/// because the [`Diagnostic`] type owns it transitively — the bin layer
/// and integration tests inspect the diagnostic without depending on the
/// engine's helper module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BurstHelper {
    /// `Engine::start_seed_burst` — Idle → `Active(PreFire(Seed))`.
    StartSeedBurst,
    /// `Engine::start_standard_burst` — Idle → `Active(PreFire(Standard))`.
    StartStandardBurst,
    /// `Engine::event_drives_batching` — pre-fire FsEvent absorb.
    EventDrivesBatching,
    /// `Engine::unstable_response_drives_batching` — unstable verify
    /// response re-arms Batching.
    UnstableResponseDrivesBatching,
    /// `Engine::transition_to_verifying` — Batching/Draining → Verifying.
    TransitionToVerifying,
    /// `Engine::transition_to_draining` — Verifying → Draining.
    TransitionToDraining,
    /// `Engine::transition_to_awaiting` — fire transition
    /// (PreFire → PostFire).
    TransitionToAwaiting,
    /// `Engine::transition_to_rebasing` — Awaiting → Rebasing.
    TransitionToRebasing,
    /// `Engine::absorb_event_into_fire_tail` — post-fire FsEvent absorb.
    AbsorbEventIntoFireTail,
    /// `Engine::restart_burst_from_fire_tail_residual` — post-rebase
    /// residual restart (`Active(PostFire)` → `Active(PreFire(Batching))`
    /// typed move; the suppress / dirty-cascade contributions stay held
    /// from the original burst start).
    RestartBurstFromFireTailResidual,
}

/// Failure mode for an [`Diagnostic::LcaIntegrityViolation`] emission.
///
/// The engine's LCA reduction (`burst::lca_target` →  `burst::lca_pair`)
/// walks each `dirty_resources` entry up the Tree to its joint
/// ancestor; the variants below distinguish the two ways that walk can
/// abort.
///
/// In production each is structurally unreachable — the upstream
/// `live` filter drops stale ids before `lca_pair` runs, and ancestry
/// walks bottom out at the FS-root scaffold — but instrumenting both
/// keeps the failure surface visible if a future refactor breaks
/// either invariant.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LcaIntegritySource {
    /// `tree.get(id)` returned `None` at `lca_pair`'s entry — an upstream
    /// caller bypassed the `live` filter and handed a stale id through.
    /// Fresh class of bug; the silent live-filter at `lca_target` should
    /// have caught this.
    StaleId,
    /// `tree.parent(id)` returned `None` mid-walk — the candidate's
    /// ancestor chain ran out before two candidates aligned. Indicates a
    /// structural break in the Tree (a node whose `parent` pointer
    /// dangles) or a depth-equalisation error.
    BrokenAncestry,
}

/// Structural cause behind a [`Diagnostic::SpliceCrossedUncovered`]
/// emission.
///
/// Mirrors the cause-tag precedent set by [`LcaIntegritySource`] —
/// both demux otherwise-defensive diagnostics into operator-actionable
/// classes without the operator having to re-trace the failure site.
/// One variant per structural failure mode inside `splice_dir_prior` /
/// `splice_dir`:
///
/// - [`Self::TargetOutsideAnchorSubtree`] — the parent walk from
///   `target` did not reach the anchor (`ancestor_chain` bottomed out).
///   Indicates a stale `ResourceId` upstream or a coverage contraction
///   that revoked the target's covering Profile.
/// - [`Self::SlotReapedMidGraft`] — an interior segment's
///   `Tree::name(next_id)` returned `None`. The slot's generation
///   advanced between burst start and graft commit (the Resource was
///   reaped under another Profile's pass).
/// - [`Self::IntermediateUncovered`] — an interior segment was stored
///   as [`crate::DirChild::Uncovered`] (or absent, or as a [`crate::ChildEntry::Leaf`])
///   in the prior snapshot. The walker recorded the slot's identity but
///   did not recurse, so the splice path cannot navigate through it.
///
/// After the walker-race fix that eliminated transient-IO `Uncovered`
/// emissions, [`Self::IntermediateUncovered`] is the only variant
/// reachable through legitimate filesystem state — and only via cross-
/// filesystem boundaries, where the walker stores an intermediate as
/// `Uncovered` because its `cmeta.dev()` differs from the anchor's
/// `root_dev`. The other two remain v1-unreachable; reaching them
/// signals an upstream LCA / Tree lifecycle regression.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpliceFailureCause {
    TargetOutsideAnchorSubtree,
    SlotReapedMidGraft,
    IntermediateUncovered,
}

/// Subject of a [`Diagnostic::PromoterClaimPurged`] emission.
///
/// Promoter claims are a disjoint set from Profile claims; the
/// dedicated enum lets operators distinguish the source without
/// parsing the embedded id type. Each claim type has a dedicated
/// bookkeeping field on [`crate::promoter::Promoter`]:
/// - [`Self::DescentPrefix`] ⇔ `Promoter.state == PrefixPending(_)`
///   (the literal-prefix descent watch).
/// - [`Self::ActiveProxy`] ⇔ `Promoter.state == Active { proxies }` and
///   `proxies.contains_key(&resource)` (one of the per-pattern proxy
///   watches).
/// - [`Self::PrefixParent`] ⇔ `Promoter.prefix_parent == Some(resource)`
///   (the preserved terminus-parent recovery edge — the Promoter twin
///   of [`ClaimKind::WatchRootParent`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PromoterClaimKind {
    DescentPrefix,
    ActiveProxy,
    PrefixParent,
}

/// Engine-emitted diagnostic. Equality is structural so tests can pin the
/// exact variant + fields produced by a given dropped Input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Diagnostic {
    /// `ProbeResponse` whose `(owner, correlation)` doesn't match the
    /// owner's in-flight `ProbeSlot` correlation. Catches stale-id
    /// (post-detach), post-cancel arrivals, and out-of-order responses across all
    /// owner kinds. The `owner` field carries the [`ProbeOwner`] so
    /// operators can demux which entity (Profile in v1) saw the stale
    /// response.
    StaleProbeResponse {
        owner: ProbeOwner,
        correlation: ProbeCorrelation,
    },
    /// `TimerExpired(id)` whose `TimerId` is not referenced by any Profile's
    /// burst. `pop_expired` already drops these silently; the variant is the
    /// defense-in-depth signal for a direct `step(Input::TimerExpired)` call
    /// from a misbehaving caller.
    StaleTimer { id: TimerId },
    /// `EffectComplete` arrived for a Profile not in
    /// [`crate::PostFirePhase::Awaiting`]. Two paths reach here legitimately:
    ///
    /// - `gate_deadline` expired and force-transitioned to `Rebasing`
    ///   (or, post-rebase, to Idle); a late completion arrives.
    /// - `finalize_anchor_lost` dropped the burst mid-`Awaiting`;
    ///   already-spawned actuator commands run to completion and report
    ///   back even though the engine no longer tracks them.
    ///
    /// In both cases the engine drops the completion (no per-Profile
    /// state to update — the burst is already over) and emits this
    /// Diagnostic so operators can see the late arrival.
    EffectCompleteOutsideAwaiting { sub: SubId, profile: ProfileId },
    /// `EffectComplete` for a Sub not in the registry. Emitted by
    /// `on_effect_complete` when the actuator delivers a completion for a
    /// `SubId` already removed from the engine. Distinct from
    /// [`Self::DetachUnknownSub`] (a stale-id detach attempt) — the
    /// triggering input is an `EffectComplete`, not a detach request.
    EffectCompleteForUnknownSub { sub: SubId },
    /// [`crate::Input::DetachSub`] targeted a `SubId` not in the
    /// registry — an external caller submitted a stale id. Hot reload
    /// no longer reaches here: `Input::ConfigDiff` resolves operator
    /// names to ids through the registry's own `by_name` index, so an
    /// unresolved `removed` name surfaces as
    /// [`Self::ConfigDiffUnknownSub`] instead. Distinct from
    /// [`Self::EffectCompleteForUnknownSub`] — that variant fires on a
    /// stray completion arrival, not on a detach miss.
    DetachUnknownSub { sub: SubId },
    /// `Input::ConfigDiff`'s Sub `removed` list named a watch the
    /// engine has no record of — typically a name whose prior attach
    /// failed ([`Self::AttachPathInvalid`]) so it never entered the
    /// registry. Benign and informational: there is nothing to detach.
    /// The resolution shim emits this rather than attempting a detach.
    ConfigDiffUnknownSub { name: CompactString },
    /// Promoter-side twin of [`Self::ConfigDiffUnknownSub`]:
    /// `Input::ConfigDiff`'s Promoter `removed` list named a Promoter
    /// the engine has no record of. Benign and informational: there is
    /// nothing to reap.
    ConfigDiffUnknownPromoter { name: CompactString },
    /// Probe returned `Vanished` during a `Seed` or `Standard` burst. The
    /// Engine's response differs by intent; the variant preserves the intent
    /// for log readability.
    ProbeVanished {
        profile: ProfileId,
        intent: BurstIntent,
    },
    /// Probe returned `Failed { errno }`. Treated identically to `Vanished`;
    /// the variant preserves errno + intent.
    ProbeFailed {
        profile: ProfileId,
        intent: BurstIntent,
        errno: i32,
    },
    /// `FsEvent` arrived for a covered descendant whose class (per the
    /// `fs_event_to_class` mapping) is not in the covering Profile's
    /// `events_union`. The user opted out of this class via `Sub.events`,
    /// so the engine drops the event before it can drive a burst (the
    /// class filter sits before dirty-set bumps).
    ///
    /// Distinct from [`Self::EventNoConsumer`] (no covering Profile at
    /// all): there *is* a covering Profile, but the class filter rejects.
    /// Distinct from the prior v1 `MetadataChangedIgnored`: that variant
    /// hard-coded "always drop METADATA"; this one carries the dropped
    /// event + Profile so logs can disambiguate user opt-out from race.
    ///
    /// Anchor-on-Profile events bypass this filter unconditionally —
    /// lifecycle continuity (anchor terminal events drive
    /// `on_anchor_terminal_event`; non-terminal anchor events drive the
    /// burst) trumps the user's class opt-out.
    EventClassDropped {
        resource: ResourceId,
        event: FsEvent,
        profile: ProfileId,
    },
    /// `FsEvent` arrived for a Resource whose `watch_demand == 0` — race
    /// between `Unwatch` op and the Sensor's kqueue drain. True "stale FD"
    /// race; the engine cannot route this event anywhere.
    EventOnUnwatchedResource { resource: ResourceId },
    /// `FsEvent` arrived for a Resource that is genuinely Watched
    /// (`watch_demand > 0`) but no Profile (or descent, or recovery
    /// target) consumed it this step — typically a `WatchRootParent`
    /// firing `StructureChanged` for an entry the engine doesn't track.
    /// Logged at TRACE: this is benign normal-operation noise, not a
    /// race or bug.
    EventNoConsumer { resource: ResourceId },
    /// `Input::WatchOpRejected` arrived from the Sensor. Engine clamped
    /// `watch_demand := 0` on `resource` and emitted no further `Watch` op
    /// for it — recovery is on the parent's next `StructureChanged` event
    /// via the `created` reconciliation. Variants where `watch_demand == 0`
    /// already (the Sensor's queue race) are reported here too with no state
    /// mutation.
    ///
    /// `failure` carries the typed kernel-error class (Pressure / Resource
    /// / Invariant) — operators read the variant directly without
    /// translating errno values per-platform.
    WatchOpRejected {
        resource: ResourceId,
        failure: WatchFailure,
    },
    /// Pending-path descent probe returned `Vanished` for `prefix`.
    /// The Engine rewinds descent to the next-existing ancestor of `prefix`.
    /// Repeated occurrences during scaffold tear-down are normal.
    PendingPathProbeVanished {
        profile: ProfileId,
        prefix: ResourceId,
    },
    /// Pending-path descent probe returned `Failed { errno }` for
    /// `prefix`. The Engine retains the pending state and waits for the
    /// next event at `prefix` (`on_descent_event`) before retrying.
    PendingPathProbeFailed {
        profile: ProfileId,
        prefix: ResourceId,
        errno: i32,
    },
    /// A Profile's active burst carried [`crate::BurstFinish::Reap`]
    /// (the last Sub had detached mid-burst), then a fresh `attach_sub`
    /// arrived at the same `(resource, config_hash)` before the burst
    /// completed — the directive is flipped back to
    /// [`crate::BurstFinish::ReturnToIdle`] and the Profile remains
    /// alive under the new Sub set. Informational — not an error.
    /// Pairs with [`Self::ProfileReaped`] (the reap actually ran).
    ReapPendingCancelled { profile: ProfileId },
    /// A Profile was reaped — its [`crate::ProfileMap`] entry is gone,
    /// every watch contribution released. `via` distinguishes the
    /// trigger so operators can tell a steady-state detach
    /// ([`crate::ReapTrigger::Immediate`]) from a deferred burst-end
    /// reap ([`crate::ReapTrigger::DeferredFromBurst`], honouring a
    /// prior [`crate::BurstFinish::Reap`]). Informational — not an
    /// error.
    ProfileReaped {
        profile: ProfileId,
        via: crate::ReapTrigger,
    },
    /// A Profile's claim on `resource` was purged because the kernel
    /// rejected the watch on it (`Input::WatchOpRejected` arrived,
    /// clamping `watch_demand := 0`). One emission per affected
    /// (Profile, claim_kind) pair — a single rejection at a multi-claim
    /// resource (anchor of P, watch-root-parent of Q, descent prefix of
    /// R) emits three.
    ///
    /// - [`ClaimKind::Anchor`]: the Profile lost its anchor watch. The
    ///   engine cancels any in-flight burst probe and finishes the
    ///   burst to Idle. Recovery is via the `watch_root_parent`'s next
    ///   `StructureChanged` (if still watched) or operator restart.
    /// - [`ClaimKind::WatchRootParent`]: the Profile loses its
    ///   parent-edge recovery channel. Anchor stays watched (different
    ///   `resource`); auto-recovery on rename/recreation is no longer
    ///   possible.
    /// - [`ClaimKind::DescentPrefix`]: the descent is abandoned. The
    ///   engine cancels any in-flight descent probe and transitions the
    ///   Profile to Idle. Recovery is operator-driven (re-attach via
    ///   SIGHUP) or, if a parent ancestor is itself watched, automatic
    ///   via the next reconciliation.
    ProfileClaimPurged {
        profile: ProfileId,
        claim: ClaimKind,
        resource: ResourceId,
        failure: WatchFailure,
    },
    /// A Promoter's claim on `resource` was purged because the kernel
    /// rejected the watch on it (`Input::WatchOpRejected` arrived,
    /// clamping `watch_demand := 0`). One emission per affected
    /// (Promoter, claim_kind) pair — a single rejection at a resource
    /// claimed by both a Profile (via anchor / watch-root-parent /
    /// descent-prefix) AND one or more Promoters (via descent-prefix
    /// or proxy) emits one [`Self::ProfileClaimPurged`] per Profile
    /// claim and one [`Self::PromoterClaimPurged`] per Promoter claim.
    ///
    /// - [`PromoterClaimKind::DescentPrefix`]: the literal-prefix
    ///   descent is abandoned. The engine cancels any in-flight
    ///   descent probe and transitions the Promoter to `Active{empty}`.
    ///   The Promoter is stranded — there is no v1 recovery channel
    ///   for the literal prefix; operator restart is required.
    /// - [`PromoterClaimKind::ActiveProxy`]: the proxy is unregistered
    ///   and any in-flight enumeration probe targeting it is cancelled.
    ///   The proxy will not re-register until a fresh enumeration of
    ///   its parent re-discovers the entry.
    /// - [`PromoterClaimKind::PrefixParent`]: the Promoter loses its
    ///   terminus-parent recovery edge (the twin of
    ///   [`ClaimKind::WatchRootParent`]). Proxies stay watched
    ///   (different `resource`); auto-recovery on terminus recreation
    ///   is no longer possible — operator restart is required.
    PromoterClaimPurged {
        promoter: PromoterId,
        claim: PromoterClaimKind,
        resource: ResourceId,
        failure: WatchFailure,
    },
    /// A path-based attach request carried a malformed path — empty,
    /// containing `.` / `..`, or a Windows prefix (unsupported on Unix
    /// v1). The engine drops the attach; no Sub is registered. `path`
    /// is the offending request and `hint` the rejection reason, so an
    /// operator submitting a hot-reload batch can identify the bad
    /// entry without re-scanning the config.
    AttachPathInvalid { path: Arc<Path>, hint: &'static str },
    /// A resource-anchored attach request named a [`ResourceId`] with no
    /// live Tree slot (reaped, never-existed, or a default sentinel).
    /// The engine drops the attach and surfaces the offending id rather
    /// than trusting the stale claim and panicking downstream — the
    /// resource-arm counterpart to [`Self::AttachPathInvalid`].
    AttachResourceStale { resource: ResourceId },
    /// A probe response's snapshot shape (`File` from `AnchorOk(_)` vs
    /// `Dir` from `SubtreeOk(_)`) disagrees with the Profile's cached
    /// [`crate::Profile::kind`]. Structurally unreachable in v1 (the
    /// engine types each probe request to the Profile's kind and the
    /// walker's outcome matches by construction), so an emission
    /// signals a walker regression. The engine recovers by treating
    /// the anchor as lost and re-deriving from disk.
    AnchorKindMismatch {
        profile: ProfileId,
        prior_kind: ResourceKind,
        response_kind: ResourceKind,
    },
    /// `splice` could not navigate from the prior snapshot's anchor
    /// down to `target`; `cause` demuxes the structural failure mode
    /// (see [`SpliceFailureCause`]). The engine contract is "graft
    /// only into observed subtrees", so any emission is a contract
    /// breach: the engine keeps its prior view and converges on the
    /// next probe. Only [`SpliceFailureCause::IntermediateUncovered`]
    /// is reachable through legitimate state (a cross-filesystem
    /// boundary); the other two signal an upstream LCA / Tree
    /// lifecycle regression.
    SpliceCrossedUncovered {
        profile: ProfileId,
        target: ResourceId,
        cause: SpliceFailureCause,
    },
    /// `FsEvent` arrived while the Profile was in
    /// [`crate::PostFirePhase::Awaiting`] or [`crate::PostFirePhase::Rebasing`]
    /// — the post-fire tail of a burst. The engine absorbs the event:
    /// no fresh burst, no settle re-arm, no `dirty_resources` extension.
    /// The Rebasing probe captures the disk state (including whatever
    /// triggered this event) into the new baseline, so the change is
    /// folded into the fire-cycle's terminal rebase rather than driving
    /// a second burst against an in-flight one. Informational; the
    /// event is not lost, merely deferred.
    EventAbsorbedByFireTail {
        profile: ProfileId,
        resource: ResourceId,
        event: FsEvent,
    },
    /// `AwaitGateDeadline` timer elapsed before all outstanding
    /// `EffectComplete`s arrived. Indicates the actuator likely has a
    /// hung child or a slow command; the engine force-transitions the
    /// burst to [`crate::PostFirePhase::Rebasing`] so it can re-establish
    /// a baseline against disk reality. Late completions land in
    /// [`Self::EffectCompleteOutsideAwaiting`].
    AwaitGateDeadlineElapsed {
        profile: ProfileId,
        outstanding: u32,
    },
    /// `Input::SensorOverflow` arrived — the kernel's event queue
    /// dropped record(s) over `scope` and the watcher surfaced the
    /// loss-of-trust signal. The engine reseeds every in-scope Profile
    /// against disk. One emission per overflow record; it is the
    /// engine's only "we missed events" signal, so an operator seeing
    /// it should tune the load condition (`max_queued_events`
    /// saturation, a slow actuator blocking the watcher's drain).
    SensorOverflow { scope: OverflowScope },
    /// `Input::SensorOverflow` arrived from the Sensor; the engine
    /// reseeded `promoter` (alongside any in-scope Profiles surfaced by
    /// [`Self::SensorOverflow`]'s peer reseed loop). Per-Promoter
    /// dispatch is state-keyed:
    /// - `PrefixPending(_)` ⇒ a fresh descent probe is emitted at
    ///   `current_prefix` (gated on the Promoter's descent slot being
    ///   unarmed; an in-flight descent probe's response will reflect
    ///   the post-overflow state).
    /// - `Active { proxies }` ⇒ every proxy is enqueued into
    ///   `pending_enumerations`; the dispatcher drains one immediately
    ///   into a probe, with the rest queued behind the single-slot.
    ///
    /// One emission per affected Promoter. The bursts the reseed
    /// schedules carry no per-Promoter annotation that they were
    /// triggered by overflow rather than a normal `FsEvent`.
    PromoterReseededForOverflow { promoter: PromoterId },
    /// A `PerStableFile` Sub's loss-window reactions were dropped: a
    /// recovery reseed absorbed the change into the rebased baseline
    /// and the per-file path keeps no survival witness (a v1
    /// limitation). Emitted once per recovery with real drift and ≥1
    /// `PerStableFile` Sub. Informational — the dropped reactions
    /// cannot be reconstructed.
    PerFileDriftDroppedOnRecovery { profile: ProfileId },
    /// A Sub has been registered with the engine and assigned `sub`.
    /// Emitted by `attach_sub_inner` on every successful insert —
    /// static (operator-declared) attaches and dynamic Promoter-spawned
    /// attaches alike.
    ///
    /// Pure operator narration: the bin logs it (INFO for static, DEBUG
    /// for dynamic). Hot-reload identity resolution does *not* route
    /// through this stream — name → `SubId` is the engine's own
    /// authoritative `by_name` index, resolved registry-side. Tests
    /// read it via `testkit::first_attached_sub` to capture the minted
    /// id.
    ///
    /// `name` carries the Sub's user-facing name verbatim — for static
    /// Subs the operator's `[[watch]].name`; for dynamic Subs the
    /// engine's synthesized `<promoter_name>@<resolved_path>` shape.
    /// `source_promoter` distinguishes the two.
    SubAttached {
        sub: SubId,
        name: CompactString,
        source_promoter: Option<PromoterId>,
    },
    /// A Promoter has been registered with the engine and assigned
    /// `promoter`. Emitted by `attach_promoter_inner`. Pure operator
    /// narration (the bin logs it at INFO); `name` carries the
    /// operator-facing Promoter name verbatim. Hot-reload resolution
    /// uses the engine's own `by_name` index, not this stream.
    PromoterAttached {
        promoter: PromoterId,
        name: CompactString,
    },
    /// A Promoter has been removed from the engine. Pairs with
    /// [`Self::PromoterAttached`]; pure operator narration (the bin
    /// logs it at INFO).
    PromoterReaped { promoter: PromoterId },
    /// Promoter literal-prefix descent probe returned `Vanished` for
    /// `prefix`. The engine rewinds descent to the next-existing
    /// ancestor of `prefix`. Repeated occurrences during scaffold
    /// tear-down are normal.
    PromoterDescentVanished {
        promoter: PromoterId,
        prefix: ResourceId,
    },
    /// Promoter literal-prefix descent probe returned `Failed { errno }`
    /// for `prefix`. The engine retains the `PrefixPending` state and
    /// awaits the next event at `prefix` before retrying.
    PromoterDescentFailed {
        promoter: PromoterId,
        prefix: ResourceId,
        errno: i32,
    },
    /// Promoter enumeration matched `path` and the engine minted a
    /// dynamic Sub for it; `kind` is the snapshot's kind for the
    /// matched entry. Operator narration — the bin logs it as
    /// "promotion observed".
    PromotionKindObserved {
        promoter: PromoterId,
        path: Arc<Path>,
        kind: ResourceKind,
    },
    /// The Promoter's live dynamic-Sub count (derived from
    /// `SubRegistry`) crossed a threshold for the first time. Operator
    /// signal that the pattern is matching more targets than typical —
    /// likely a too-broad pattern (e.g. `/*` without further
    /// constraint). One-shot per Promoter lifetime; the latch on
    /// `Promoter.warned_at_threshold` suppresses repeats.
    PromoterFanoutThreshold { promoter: PromoterId, count: usize },
    /// `FsEvent` arrived for a Resource that previously held a
    /// `proxy_promoters` back-ref to `promoter`, but the Promoter has
    /// either reaped the proxy or fully reaped during the same step.
    /// Engine drops the event; operators can ignore. Pairs with
    /// [`Self::EventNoConsumer`] — the proxy back-ref was the
    /// supposed consumer; the back-ref is now stale.
    PromoterProxyStaleEvent {
        promoter: PromoterId,
        resource: ResourceId,
    },
    /// Promoter enumeration probe at `proxy` returned `Vanished` — the
    /// proxy directory is gone from disk. The engine cascade-cleans the
    /// proxy and any sub-proxies rooted under it via
    /// `unregister_proxy_subtree`; dynamic Subs anchored at or below
    /// `proxy` reap via their own anchor-terminal events. Distinct from
    /// [`Self::PromoterDescentVanished`] — that variant fires during
    /// the literal-prefix descent (`PrefixPending` state); this one
    /// during the proxy-fanout enumeration (`Active` state).
    PromoterEnumerationVanished {
        promoter: PromoterId,
        proxy: ResourceId,
    },
    /// Promoter enumeration probe at `proxy` returned `Failed { errno }`.
    /// The engine retains proxy state; the next event at `proxy` will
    /// re-trigger enumeration. Typical errnos are transient (`EACCES`,
    /// `EIO`); a permanent failure leaves the proxy stalled until the
    /// underlying condition clears or the operator restarts.
    PromoterEnumerationFailed {
        promoter: PromoterId,
        proxy: ResourceId,
        errno: i32,
    },
    /// A dynamic Sub minted by `promoter` at `path` was reaped because
    /// its anchor disappeared. Operator narration; if the path
    /// re-materialises a fresh enumeration re-promotes it.
    DynamicSubReaped {
        promoter: PromoterId,
        sub: SubId,
        path: Arc<Path>,
    },
    /// A burst-lifecycle helper was invoked on a Profile whose state
    /// did not match the helper's typed precondition (e.g.,
    /// `transition_to_verifying` called on a Profile in
    /// `ActivePostFire`). The helper bails before mutating any
    /// state; this variant surfaces the routing breach so the
    /// originating dispatcher can be fixed.
    ///
    /// `helper` names which entry point bailed; `observed` reports
    /// the Profile's state at the call. Stale `ProfileId`
    /// (`profiles.get(_) == None`) does NOT emit this variant — that
    /// is a benign post-detach race, surfaced only when an op
    /// targeting the missing slot rises through the usual handlers.
    ///
    /// Structurally unreachable in v1 — every dispatcher gates on the
    /// state variant before reaching a helper — but the precondition
    /// gates the gate, so any future routing regression surfaces here
    /// instead of silently dropping the transition.
    InvalidBurstTransition {
        profile: ProfileId,
        helper: BurstHelper,
        observed: ProfileStateDiscriminant,
    },
    /// The engine's LCA reduction over a burst's `dirty_resources`
    /// failed an internal integrity check. `source` distinguishes the
    /// two failure modes; the caller (`lca_target`) recovers by
    /// folding the joint candidate to anchor so the burst still has a
    /// probe target — the diagnostic surfaces the breach without
    /// stalling the lifecycle.
    ///
    /// See [`LcaIntegritySource`] for the two failure modes. Both are
    /// structurally unreachable in v1 (the upstream `live` filter
    /// catches stale ids; Tree ancestry walks always bottom out at the
    /// FS-root scaffold), so an emission here is a regression signal,
    /// not noise.
    LcaIntegrityViolation {
        profile: ProfileId,
        source: LcaIntegritySource,
    },
}
