//! Engine diagnostics.
//!
//! Emitted alongside the dropped/clamped Inputs they explain. Variant set grows phase-by-phase as
//! new drop paths land. Each variant is light-weight (a few small fields) and carries enough
//! context to log meaningfully.

use crate::ids::{ProbeCorrelation, ProfileId, ResourceId, SubId, TimerId};
use crate::input::{FsEvent, OverflowScope};
use crate::op::{ProbeFailure, WatchFailure};
use crate::profile::{AbsorbMode, BurstIntent, ProfileStateDiscriminant};
use crate::resource::ResourceKind;
use compact_str::CompactString;
use std::path::Path;
use std::sync::Arc;

/// Which Profile-side claim was the subject of a [`Diagnostic::ProfileClaimPurged`] emission. Each
/// claim type has a dedicated bookkeeping field on [`crate::profile::Profile`]:
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
/// Tagged onto [`Diagnostic::InvalidBurstTransition`]. Each variant names exactly one helper in
/// `specter-engine`'s `burst.rs`. Variants are added when a helper is created; helpers without a
/// typed precondition (the idempotent `finish_burst_to_idle`) are absent by design.
///
/// The enum is exported from `specter-core` rather than `specter-engine` because the [`Diagnostic`]
/// type owns it transitively — the bin layer and integration tests inspect the diagnostic without
/// depending on the engine's helper module.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BurstHelper {
    /// `Engine::start_seed_burst` — Idle → `Active(PreFire(Seed))`.
    StartSeedBurst,
    /// `Engine::start_standard_burst` — Idle → `Active(PreFire(Standard))`.
    StartStandardBurst,
    /// `Engine::event_drives_batching` — pre-fire FsEvent absorb.
    EventDrivesBatching,
    /// `Engine::retry_drives_batching` — a [`crate::QuiescenceVerdict::Retry`] verify response
    /// (channel disagreement or transient non-observation with the ceiling not yet fired) re-arms
    /// Batching for the next sample.
    RetryDrivesBatching,
    /// `Engine::transition_to_verifying` — Batching/Draining → Verifying.
    TransitionToVerifying,
    /// `Engine::transition_to_draining` — Verifying → Draining.
    TransitionToDraining,
    /// `Engine::transition_to_awaiting` — fire transition (PreFire → PostFire).
    TransitionToAwaiting,
    /// `Engine::transition_to_rebasing` — Awaiting → Rebasing (natural post-`EffectComplete`
    /// unforced, or gate-deadline-recovery forced) or Settling → Rebasing (settle-expiry or
    /// ceiling-driven force, out of the HashChannel spacing loop). Rebase-loop ceiling arming lives
    /// separately on `Engine::arm_rebase_loop_ceiling` (the natural `Awaiting → Rebasing` entry
    /// only); this helper is single-purpose (mint correlation, clear residual, write phase, emit
    /// probe).
    TransitionToRebasing,
    /// `Engine::transition_to_settling` — post-fire HashChannel re-sample spacing. Reached only
    /// from the `Rebasing → Settling` retry loop-back (`dispatch_rebase_ok::Retry`); the post-fire
    /// mirror of pre-fire's `retry_drives_batching`. The natural rebase entry is probe-first
    /// (`Awaiting → Rebasing`), so there is no `Awaiting → Settling` advance.
    TransitionToSettling,
    /// `Engine::absorb_event_into_fire_tail` — post-fire FsEvent absorb.
    AbsorbEventIntoFireTail,
    /// `Engine::restart_burst_from_fire_tail_residual` — post-rebase residual restart
    /// (`Active(PostFire)` → `Active(PreFire(Batching))` typed move; the watched anchor is
    /// preserved across the move, no refcount edge changes).
    RestartBurstFromFireTailResidual,
}

/// Structural cause behind a [`Diagnostic::SpliceCrossedUncovered`] emission.
///
/// Demuxes an otherwise-defensive diagnostic into operator-actionable classes without the operator
/// having to re-trace the failure site. One variant per structural failure mode inside
/// `splice_dir_prior` / `splice_dir`:
///
/// - [`Self::TargetOutsideAnchorSubtree`] — the parent walk from `target` did not reach the anchor
///   (`ancestor_chain` bottomed out). Indicates a stale `ResourceId` upstream or a coverage
///   contraction that revoked the target's covering Profile.
/// - [`Self::SlotReapedMidGraft`] — an interior segment's `Tree::name(next_id)` returned `None`.
///   The slot's generation advanced between burst start and graft commit (the Resource was reaped
///   under another Profile's pass).
/// - [`Self::IntermediateUncovered`] — an interior segment was stored as
///   [`crate::DirChild::Uncovered`] (or absent, or as a [`crate::ChildEntry::Leaf`]) in the prior
///   snapshot. The walker recorded the slot's identity but did not recurse, so the splice path
///   cannot navigate through it.
///
/// After the walker-race fix that eliminated transient-IO `Uncovered` emissions,
/// [`Self::IntermediateUncovered`] is the only variant reachable through legitimate filesystem
/// state — and only via cross- filesystem boundaries, where the walker stores an intermediate as
/// `Uncovered` because its `cmeta.dev()` differs from the anchor's `root_dev`. The other two remain
/// v1-unreachable; reaching them signals an upstream LCA / Tree lifecycle regression.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SpliceFailureCause {
    TargetOutsideAnchorSubtree,
    SlotReapedMidGraft,
    IntermediateUncovered,
}

/// Why a Sub was detached from the engine.
///
/// Carried on [`Diagnostic::SubDetached`] so operators can distinguish lifecycle paths without
/// inferring the trigger from surrounding diagnostics. Each variant names exactly one detach origin
/// — adding one to the engine's detach surface is a compile error here.
///
/// - [`Self::ConfigDiffRemoved`]: the operator removed the `[[watch]]` block from config;
///   hot-reload's `subs.removed` arm in `on_config_diff` drives the detach via `detach_sub_inner`.
/// - [`Self::ConfigDiffIdentityChanged`]: the operator changed the Sub's identity (anchor / scan
///   config / `max_settle` / events) in config; the diff's `modified_identity` bucket routes
///   through detach + attach, so the same operator name briefly leaves the registry. Distinct from
///   the in-place `modified_params` arm (which emits [`Diagnostic::SubRebound`] without any
///   `SubDetached`). The bucket name is the precise term: `identity` is what changed; `name` and
///   `has_fired` continuity is what the in-place arm preserves.
/// - [`Self::IpcDisabled`]: an operator runtime-disabled the Sub via the bin's IPC `disable` verb;
///   the bin sends [`crate::Input::DetachSub`] carrying this reason verbatim.
/// - [`Self::AnchorLost`]: a discovery-minted Sub's anchor disappeared and the all-dynamic
///   anchor-terminal teardown unwound its Profile — the honest vocabulary for "the watched path is
///   gone", with no source-entity reap implied (the discovery template stays live and re-mints on
///   reappearance). Pairs with [`Diagnostic::DiscoverySubReaped`] (source-keyed, path-carrying
///   narration).
/// - [`Self::DiscoverySourceDetached`]: the discovery Sub that minted this Sub was detached, so the
///   cascade reaped the minted set — named for what actually happened to *this* Sub's source
///   rather than overloading the anchor story.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum DetachReason {
    ConfigDiffRemoved,
    ConfigDiffIdentityChanged,
    IpcDisabled,
    AnchorLost,
    DiscoverySourceDetached,
}

/// Engine-emitted diagnostic. Equality is structural so tests can pin the exact variant + fields
/// produced by a given dropped Input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Diagnostic {
    /// `ProbeResponse` whose `(owner, correlation)` doesn't match the owner's in-flight `ProbeSlot`
    /// correlation. Catches stale-id (post-detach), post-cancel arrivals, and out-of-order
    /// responses. The `owner` field carries the [`ProfileId`] so operators can see which Profile
    /// saw the stale response.
    StaleProbeResponse {
        owner: ProfileId,
        correlation: ProbeCorrelation,
    },
    /// `TimerExpired(id)` whose `TimerId` is not referenced by any Profile's burst. `pop_expired`
    /// already drops these silently; the variant is the defense-in-depth signal for a direct
    /// `step(Input::TimerExpired)` call from a misbehaving caller.
    StaleTimer { id: TimerId },
    /// `EffectComplete` arrived for a Profile not in [`crate::PostFirePhase::Awaiting`]. Two paths
    /// reach here legitimately:
    ///
    /// - `gate_deadline` expired and force-transitioned to `Rebasing` (or, post-rebase, to Idle); a
    ///   late completion arrives.
    /// - `finalize_anchor_lost` dropped the burst mid-`Awaiting`; already-spawned actuator commands
    ///   run to completion and report back even though the engine holds no state tracking them.
    ///
    /// In both cases the engine drops the completion (no per-Profile state to update — the burst is
    /// already over) and emits this Diagnostic so operators can see the late arrival.
    EffectCompleteOutsideAwaiting { sub: SubId, profile: ProfileId },
    /// `EffectComplete` for a Sub not in the registry. Emitted by `on_effect_complete` when the
    /// actuator delivers a completion for a `SubId` already removed from the engine. Distinct from
    /// [`Self::DetachUnknownSub`] (a stale-id detach attempt) — the triggering input is an
    /// `EffectComplete`, not a detach request.
    EffectCompleteForUnknownSub { sub: SubId },
    /// [`crate::Input::DetachSub`] targeted a `SubId` not in the registry — an external caller
    /// submitted a stale id. Hot reload does not reach here: `Input::ConfigDiff` resolves operator
    /// names to ids through the registry's own `by_name` index, so an unresolved `removed` name
    /// surfaces as [`Self::ConfigDiffUnknownSub`] instead. Distinct from
    /// [`Self::EffectCompleteForUnknownSub`] — that variant fires on a stray completion arrival,
    /// not on a detach miss.
    DetachUnknownSub { sub: SubId },
    /// `Input::ConfigDiff`'s Sub `removed` list named a watch the engine has no record of —
    /// typically a name whose prior attach failed ([`Self::AttachPathInvalid`]) so it never entered
    /// the registry. Benign and informational: there is nothing to detach. The resolution shim
    /// emits this rather than attempting a detach.
    ConfigDiffUnknownSub { name: CompactString },
    /// `Input::ConfigDiff`'s Sub `modified_params` bucket named a watch the engine has no record of
    /// — typically a name whose prior attach failed ([`Self::AttachPathInvalid`]) so it never
    /// entered the registry. The dispatcher cannot rebind a Sub that does not exist, so it degrades
    /// the entry to a fresh attach (the same effect a future operator-driven attach would have on a
    /// clean registry). This variant frames the *reason* for triage; the fallback attach emits its
    /// own lifecycle diagnostics ([`Self::SubAttached`] on success, [`Self::AttachPathInvalid`] /
    /// [`Self::AttachResourceStale`] on failure) independently.
    ConfigDiffRebindFallbackAttach { name: CompactString },
    /// Probe returned `Vanished` during a `Seed` or `Standard` burst. The Engine's response differs
    /// by intent; the variant preserves the intent for log readability.
    ProbeVanished {
        profile: ProfileId,
        intent: BurstIntent,
    },
    /// Probe returned [`crate::op::ProbeOutcome::Failed`]. Treated identically to `Vanished`; the
    /// variant preserves the typed [`ProbeFailure`] routing target + intent. Operator-visible errno
    /// reads off `failure.errno()` at the IPC seam.
    ProbeFailed {
        profile: ProfileId,
        intent: BurstIntent,
        failure: ProbeFailure,
    },
    /// `FsEvent` arrived for a covered descendant whose class (per the `fs_event_to_class` mapping)
    /// is not in the covering Profile's `events_union`. The user opted out of this class via
    /// `Sub.events`, so the engine drops the event before it can drive a burst (the class filter
    /// sits before dirty-set bumps).
    ///
    /// Distinct from [`Self::EventNoConsumer`] (no covering Profile at all): there *is* a covering
    /// Profile, but the class filter rejects. Distinct from the prior v1 `MetadataChangedIgnored`:
    /// that variant hard-coded "always drop METADATA"; this one carries the dropped event + Profile
    /// so logs can disambiguate user opt-out from race.
    ///
    /// Anchor-on-Profile events bypass this filter unconditionally — lifecycle continuity (anchor
    /// terminal events drive `on_anchor_terminal_event`; non-terminal anchor events drive the
    /// burst) trumps the user's class opt-out.
    EventClassDropped {
        resource: ResourceId,
        event: FsEvent,
        profile: ProfileId,
    },
    /// `FsEvent` arrived for a Resource whose `watch_demand == 0` — race between `Unwatch` op and the
    /// Sensor's kqueue drain. True "stale FD" race; the engine cannot route this event anywhere.
    EventOnUnwatchedResource { resource: ResourceId },
    /// `FsEvent` arrived for a Resource that is genuinely Watched (`watch_demand > 0`) but no
    /// Profile (or descent, or recovery target) consumed it this step — typically a
    /// `WatchRootParent` firing `StructureChanged` for an entry the engine doesn't track. Logged at
    /// TRACE: this is benign normal-operation noise, not a race or bug.
    EventNoConsumer { resource: ResourceId },
    /// `Input::WatchOpRejected` arrived from the Sensor. Engine clamped `watch_demand := 0` on
    /// `resource` and emitted no further `Watch` op for it — recovery is on the parent's next
    /// `StructureChanged` event via the `created` reconciliation. Variants where `watch_demand ==
    /// 0` already (the Sensor's queue race) are reported here too with no state mutation.
    ///
    /// `failure` carries the typed kernel-error class (Pressure / Resource / Invariant) — operators
    /// read the variant directly without translating errno values per-platform.
    WatchOpRejected {
        resource: ResourceId,
        failure: WatchFailure,
    },
    /// Pending-path descent probe returned `Vanished` for `prefix`. The Engine rewinds descent to the
    /// next-existing ancestor of `prefix`. Repeated occurrences during scaffold tear-down are normal.
    PendingPathProbeVanished {
        profile: ProfileId,
        prefix: ResourceId,
    },
    /// Pending-path descent probe returned [`crate::op::ProbeOutcome::Failed`] for `prefix`. The
    /// Engine retains the pending state and waits for the next event at `prefix`
    /// (`on_descent_event`) before retrying. `failure` carries the typed routing target; the
    /// operator-visible errno reads off `failure.errno()`.
    PendingPathProbeFailed {
        profile: ProfileId,
        prefix: ResourceId,
        failure: ProbeFailure,
    },
    /// A Profile's active burst carried [`crate::BurstFinish::Reap`] (the last Sub had detached
    /// mid-burst), then a fresh `attach_sub` arrived at the same `(resource, config_hash)` before
    /// the burst completed — the directive is flipped back to [`crate::BurstFinish::ReturnToIdle`]
    /// and the Profile remains alive under the new Sub set. Informational — not an error. Pairs
    /// with [`Self::ProfileReaped`] (the reap actually ran).
    ReapPendingCancelled { profile: ProfileId },
    /// A Profile was reaped — its [`crate::ProfileMap`] entry is gone, every watch contribution
    /// released. `via` distinguishes the trigger so operators can tell a steady-state detach
    /// ([`crate::ReapTrigger::Immediate`]) from a deferred burst-end reap
    /// ([`crate::ReapTrigger::DeferredFromBurst`], honouring a prior [`crate::BurstFinish::Reap`]).
    /// Informational — not an error.
    ProfileReaped {
        profile: ProfileId,
        via: crate::ReapTrigger,
    },
    /// A Profile's claim on `resource` was purged because the kernel rejected the watch on it
    /// (`Input::WatchOpRejected` arrived, clamping `watch_demand := 0`). One emission per affected
    /// (Profile, claim_kind) pair — a single rejection at a multi-claim resource (anchor of P,
    /// watch-root-parent of Q, descent prefix of R) emits three.
    ///
    /// - [`ClaimKind::Anchor`]: the Profile lost its anchor watch. The engine cancels any in-flight
    ///   burst probe and finishes the burst to Idle. Recovery is via the `watch_root_parent`'s next
    ///   `StructureChanged` (if still watched) or operator restart.
    /// - [`ClaimKind::WatchRootParent`]: the Profile loses its parent-edge recovery channel. Anchor
    ///   stays watched (different `resource`); without the parent-edge channel, rename/recreation
    ///   cannot auto-recover — operator restart is required.
    /// - [`ClaimKind::DescentPrefix`]: the descent is abandoned. The engine cancels any in-flight
    ///   descent probe and transitions the Profile to Idle. Recovery is operator-driven (re-attach
    ///   via SIGHUP) or, if a parent ancestor is itself watched, automatic via the next
    ///   reconciliation.
    ProfileClaimPurged {
        profile: ProfileId,
        claim: ClaimKind,
        resource: ResourceId,
        failure: WatchFailure,
    },
    /// A path-based attach request carried a malformed path — empty, containing `.` / `..`, or a
    /// Windows prefix (unsupported on Unix v1). The engine drops the attach; no Sub is registered.
    /// `path` is the offending request and `hint` the rejection reason, so an operator submitting a
    /// hot-reload batch can identify the bad entry without re-scanning the config.
    AttachPathInvalid { path: Arc<Path>, hint: &'static str },
    /// A resource-anchored attach request named a [`ResourceId`] with no live Tree slot (reaped,
    /// never-existed, or a default sentinel). The engine drops the attach and surfaces the
    /// offending id rather than trusting the stale claim and panicking downstream — the
    /// resource-arm counterpart to [`Self::AttachPathInvalid`].
    AttachResourceStale { resource: ResourceId },
    /// A probe response's snapshot shape (`File` from `AnchorOk(_)` vs `Dir` from `SubtreeProven {
    /// .. }`) disagrees with the Profile's cached [`crate::Profile::kind`]. Structurally
    /// unreachable in v1 (the engine types each probe request to the Profile's kind and the
    /// walker's outcome matches by construction), so an emission signals a walker regression. The
    /// engine recovers by treating the anchor as lost and re-deriving from disk.
    AnchorKindMismatch {
        profile: ProfileId,
        prior_kind: ResourceKind,
        response_kind: ResourceKind,
    },
    /// `splice` could not navigate from the prior snapshot's anchor down to `target`; `cause` demuxes
    /// the structural failure mode (see [`SpliceFailureCause`]). The engine contract is "graft only
    /// into observed subtrees", so any emission is a contract breach: the engine keeps its prior view
    /// and converges on the next probe. Only [`SpliceFailureCause::IntermediateUncovered`] is
    /// reachable through legitimate state (a cross-filesystem boundary); the other two signal an
    /// upstream LCA / Tree lifecycle regression.
    SpliceCrossedUncovered {
        profile: ProfileId,
        target: ResourceId,
        cause: SpliceFailureCause,
    },
    /// `FsEvent` arrived while the Profile was in [`crate::PostFirePhase::Awaiting`] or
    /// [`crate::PostFirePhase::Rebasing`] — the post-fire tail of a burst. The engine absorbs the
    /// event: no fresh burst, no settle re-arm, no `dirty` extension. The Rebasing probe captures
    /// the disk state (including whatever triggered this event) into the new baseline, so the
    /// change is folded into the fire-cycle's terminal rebase rather than driving a second burst
    /// against an in-flight one. Informational; the event is not lost, merely deferred.
    EventAbsorbedByFireTail {
        profile: ProfileId,
        resource: ResourceId,
        event: FsEvent,
    },
    /// `AwaitGateDeadline` timer elapsed before all outstanding `EffectComplete`s arrived on a live
    /// (non-Reap) burst — the actuator likely has a hung child or a slow command. The engine
    /// force-transitions the burst from [`crate::PostFirePhase::Awaiting`] to
    /// [`crate::PostFirePhase::Rebasing`] so it can re-establish a baseline against disk reality.
    /// Late completions for this profile land in [`Self::EffectCompleteOutsideAwaiting`].
    ///
    /// Paired with [`Self::AwaitGateDeadlineReap`] — the same trigger on a zombie burst takes the
    /// reap path instead, and operators see distinct vocabularies for the two recoveries.
    AwaitGateDeadlineForceRebasing {
        profile: ProfileId,
        outstanding: u32,
    },
    /// `AwaitGateDeadline` timer elapsed before all outstanding `EffectComplete`s arrived on a
    /// zombie ([`crate::BurstFinish::Reap`]) burst — the only Sub detached mid-`Awaiting`, so the
    /// burst has no consumer for a rebased baseline. The engine elides the rebase round-trip
    /// (wasted work on a dying Profile) and routes through `finish_burst_to_idle`, which then
    /// dispatches the deferred reap. Late completions land in [`Self::EffectCompleteForUnknownSub`]
    /// (the Sub left the registry at detach).
    ///
    /// The structural mirror of [`Self::AwaitGateDeadlineForceRebasing`] — same gate-deadline
    /// trigger, the [`crate::BurstFinish`] in effect at expiry picks the variant.
    AwaitGateDeadlineReap {
        profile: ProfileId,
        outstanding: u32,
    },
    /// A `forced` (max-settle) verify returned a probe the walker could not fully discharge: a
    /// non-observation (an mtime-skipped or degraded frame) lies on an obligation chain at
    /// `first_unread`, so the response cannot certify quiescence. The engine refuses to act on an
    /// unprovable tree — it finishes the burst to Idle **without** committing the unread snapshot
    /// to `current` (an unread region must never become the dedup / Seed baseline) and **without**
    /// advancing the carrier's certified-sample proof, then releases the probe slot. This is a
    /// liveness terminal, not a wedge: the next `FsEvent` opens a fresh burst, and a transient
    /// cause (e.g. an `EACCES` later cleared) recovers on its own. The non-forced case retries
    /// silently within the burst's settle window and never reaches here.
    ///
    /// `intent` preserves which burst hit the ceiling — "a Seed could not establish a baseline" and
    /// "a Standard could not reconfirm" are distinct operator stories on the same terminal. The
    /// engine's consequence is identical for both (the single `undischarged_consequence` site); the
    /// field exists for log readability, exactly as on [`Self::ProbeVanished`] /
    /// [`Self::ProbeFailed`].
    ///
    /// First diagnostic to pair a [`ProfileId`] with an `Arc<Path>`: `first_unread` is the walker's
    /// path-based ledger entry (the walker has no `Tree` / [`ResourceId`]), so the path is mandated
    /// by the wire, not chosen by precedent.
    QuiescenceCeilingUnreadable {
        profile: ProfileId,
        first_unread: Arc<Path>,
        intent: BurstIntent,
    },
    /// The pre-fire `BurstDeadline` ceiling fired AND the hash channel observed concrete disagreement
    /// (`prior != response`) at the last sample: the tree was visibly still moving when the deadline
    /// expired. The engine fires/pins against the freshest observation anyway (a bounded terminal,
    /// not a wedge), exactly as on the quiet forced path — the distinction is operator-visible only.
    ///
    /// The pre-fire counterpart of [`Self::RebaseCeilingForced`]. Unlike post-fire's loud baseline,
    /// pre-fire's quiet forced-ceiling path is silent (`forced` already propagates onto
    /// `Effect.forced`, visible downstream), so only the strong-signal arm earns a diagnostic.
    ///
    /// Reachable only when the per-Profile hash channel was engaged — the burst owed quiescence
    /// proof (Standard / triggered Seed / post-recovery Seed) AND
    /// [`crate::Profile::events_witness_quiescence`] was `false` (events-incomplete mask).
    /// "Engaged," not "fired": a no-drift post-recovery Seed engages the channel yet seals via
    /// `SilentPin`, and a burst caught by an `absorb` window engages it yet commits silently
    /// ([`Self::QuiescenceAbsorbed`]) — so this is a "committed despite change" signal, independent
    /// of whether an Effect fired. For events-reliable Profiles and cold-Seed bursts the channel is
    /// bypassed and this variant is unreachable by the verdict fold (`hash_channel_disagreed` is
    /// always `false`).
    QuiescenceCeilingForcedDespiteChange {
        profile: ProfileId,
        intent: BurstIntent,
    },
    /// The post-fire rebase loop reached its `RebaseCeiling` and the engine pinned the freshest
    /// observation as the new baseline anyway (a bounded terminal, not a wedge), then finished the
    /// burst. `observed_change` is the verdict's `hash_channel_disagreed` bit — whether the hash
    /// channel observed concrete disagreement (`prior != response`) at the last `WholeSubtree`
    /// sample:
    ///
    /// - `true` — the channel was active AND disagreed: the post-command tree was visibly still
    ///   moving when the ceiling expired (the strong signal).
    /// - `false` — the ceiling expired without the hash channel observing disagreement. Reachable
    ///   shapes: settle-spaced reads agreed at the last sample (`prior == response`); the ceiling
    ///   forced the first sample before a second could confirm (`prior == None` — absence of
    ///   confirmation, not observed change); or the hash channel was inactive because the Profile's
    ///   `events_union` already witnesses quiescence ([`crate::Profile::events_witness_quiescence`]).
    ///
    /// Always emitted on the post-fire forced arm — deliberately **loud** on both bits, because no
    /// `Effect` carries `forced` downstream to record that the rebase was a ceiling fallback. This
    /// is the principled asymmetry with the pre-fire counterpart
    /// [`Self::QuiescenceCeilingForcedDespiteChange`], which emits *only* on disagreement and stays
    /// silent on the quiet path (there `forced` already rides `Effect.forced`). `intent`
    /// distinguishes a Standard post-fire rebase from a Seed-drift one, exactly as on
    /// [`Self::ProbeVanished`] / [`Self::ProbeFailed`].
    RebaseCeilingForced {
        profile: ProfileId,
        intent: BurstIntent,
        observed_change: bool,
    },
    /// The post-fire rebase loop reached its ceiling and the final `WholeSubtree` read could not
    /// discharge its obligation: a non-observation (an mtime-skipped / degraded frame) lies on an
    /// obligation chain at `first_unread`, so the response cannot certify the post-command tree.
    /// The engine refuses to rebase `baseline := current` blind — it finishes the burst **without**
    /// committing the unread snapshot and **without** advancing the rebase carrier's proof, leaving
    /// the prior baseline in place. Not a wedge: the next `FsEvent` opens a fresh burst and a
    /// transient cause (e.g. an `EACCES` later cleared) recovers on its own.
    ///
    /// The post-fire analog of [`Self::QuiescenceCeilingUnreadable`] (which is the *pre-fire*
    /// verify ceiling); same `Arc<Path>` + `intent` shape, distinct terminal so the operator story
    /// ("could not reconfirm the post-command tree" vs "could not reconfirm / establish a pre-fire
    /// baseline") survives.
    RebaseCeilingUnreadable {
        profile: ProfileId,
        first_unread: Arc<Path>,
        intent: BurstIntent,
    },
    /// `Input::SensorOverflow` arrived — the kernel's event queue dropped record(s) over `scope`
    /// and the watcher surfaced the loss-of-trust signal. The engine reseeds every in-scope Profile
    /// against disk. One emission per overflow record; it is the engine's only "we missed events"
    /// signal, so an operator seeing it should tune the load condition (`max_queued_events`
    /// saturation, a slow actuator blocking the watcher's drain).
    SensorOverflow { scope: OverflowScope },
    /// A `PerStableFile` Sub's loss-window reactions were dropped: a recovery reseed absorbed the
    /// change into the rebased baseline and the per-file path keeps no survival witness (a v1
    /// limitation). Emitted once per recovery with real drift and ≥1 `PerStableFile` Sub.
    /// Informational — the dropped reactions cannot be reconstructed.
    PerFileDriftDroppedOnRecovery { profile: ProfileId },
    /// A `PerStableFile` Sub did not fire on a fresh Profile's first-ever fire: the Seed witnessed
    /// activity (so the Profile's `SubtreeRoot` Subs fired), but a fresh Profile has no baseline, so
    /// `emit_effects` builds no per-leaf diff and the per-file reactions have nothing to enumerate.
    /// Emitted once per fresh-with-activity Seed fire that carries ≥1 `PerStableFile` Sub.
    /// Informational — running the per-file command for every file in the initial tree is never the
    /// intent; per-file reactions begin from the post-command baseline the fire establishes.
    PerFileFireSkippedOnFreshSeed { profile: ProfileId },
    /// A Sub has been registered with the engine and assigned `sub`. Emitted by `attach_sub_inner`
    /// on every successful insert — static (operator-declared) attaches and discovery-minted
    /// dynamic attaches alike.
    ///
    /// Pure operator narration: the bin logs it (INFO for static, DEBUG for dynamic). Hot-reload
    /// identity resolution does *not* route through this stream — name → `SubId` is the engine's
    /// own authoritative `by_name` index, resolved registry-side. Tests read it via
    /// `testkit::first_attached_sub` to capture the minted id.
    ///
    /// `name` carries the Sub's user-facing name verbatim — for static Subs the operator's
    /// `[[watch]].name`; for dynamic Subs the engine's synthesized
    /// `<template_name>@<matched_path>` shape. `source_discovery` distinguishes the two.
    SubAttached {
        sub: SubId,
        name: CompactString,
        source_discovery: Option<SubId>,
    },
    /// A Sub emitted [`crate::Effect`]s on this `emit_effects` pass. `count` is `1` for a
    /// `SubtreeRoot` emission, and the per-leaf emission count for a `PerStableFile` Sub — the
    /// diagnostic aggregates the pass so a one-Sub-many-files burst is one wire event, not N.
    /// Suppressed (B1-dedup) and skipped (scope-mismatch) passes emit nothing; `count > 0` is
    /// structural for every emission of this variant.
    ///
    /// The operator-facing lifecycle signal for a fire — increments the per-Sub `fire_count` /
    /// `last_fired_at` counters.
    SubFired {
        sub: SubId,
        profile: ProfileId,
        count: u32,
    },
    /// A burst folded instead of firing: an armed `absorb` window caught a would-have-fired
    /// verdict, so the engine advanced the baseline silently (the rebase-family seal) rather than
    /// running the Subs' reactions. The fold counterpart of [`Self::SubFired`] — one emission per
    /// folded episode, at the verdict floor's `AbsorbFold` arm.
    ///
    /// Carries **no hash**: the metadata hash is meaningless across machines, and folding a remote
    /// replication is the whole point. `profile`-scoped, not `sub`-scoped, because a fold is
    /// per-Profile — every Sub on the Profile folds together. Bumps the per-Profile `absorb_count`.
    ///
    /// On a transfer longer than `max_settle` the forced ceiling can emit
    /// [`Self::QuiescenceCeilingForcedDespiteChange`] *alongside* this — both are truthful
    /// ("committed despite change" + "folded"), not contradictory.
    QuiescenceAbsorbed { profile: ProfileId },
    /// An operator armed an `absorb` window on a Profile (the [`crate::Input::ArmAbsorb`] handler).
    /// Emitted once per arm, so a `tail` sees the *arm*, not only the eventual
    /// [`Self::QuiescenceAbsorbed`] fold. `mode` distinguishes a one-shot consume-on-first window
    /// from a time-boxed persist window; the expiry instant is **not** carried (an `Instant` has no
    /// clean wire wall-clock at this layer — `show` renders the expiry from live Profile state
    /// instead).
    AbsorbArmed {
        profile: ProfileId,
        mode: AbsorbMode,
    },
    /// A Sub was removed from the engine; `reason` demuxes the origin. Emitted once per Sub removal
    /// — the operator-facing lifecycle signal that complements [`Self::SubAttached`] /
    /// [`Self::SubRebound`].
    ///
    /// Distinct from siblings:
    /// - [`Self::DetachUnknownSub`] — a *failed* detach (stale id); no Sub was removed, no
    ///   `SubDetached`.
    /// - [`Self::DiscoverySubReaped`] — source-keyed narration for the same dynamic-Sub teardown
    ///   that this variant captures per-Sub. The two pair: `DiscoverySubReaped` carries the path,
    ///   `SubDetached` carries the per-Sub `(sub, profile, reason)` triple.
    /// - [`Self::SubRebound`] — in-place `modified_params` rebind; `has_fired` and the per-Sub
    ///   history survive, no `SubDetached`. The `modified_identity` arm IS a detach + attach and
    ///   DOES emit `SubDetached` with [`DetachReason::ConfigDiffIdentityChanged`].
    SubDetached {
        sub: SubId,
        profile: ProfileId,
        reason: DetachReason,
    },
    /// A Sub's per-Sub fields (`program`, `scope`, `settle`, `log_output`) were rebound in place
    /// via `rebind_sub_inner` — the `modified_params` arm of [`crate::SubRegistryDiff`].
    /// Symmetric with [`Self::SubAttached`]; pure operator narration.
    ///
    /// **`has_fired` is preserved across rebind.** The B1 dedup floor reads it as "this Sub has
    /// already announced the current stable tree state"; a program swap changes *what runs*, not
    /// *whether the tree changed*. The next event-driven burst picks up the new program; the next
    /// `seed_drift_observed` picks up the new scope and `needs_diff`. Operators who specifically
    /// want a re-fire after a program swap can restart Specter.
    ///
    /// No `name` field: identity is by id, and the rebind invariant guarantees the prior
    /// `SubAttached`'s name is still in force.
    SubRebound { sub: SubId },
    /// `rebind_sub_inner` was invoked with a stale [`SubId`]. The invariant is that the dispatcher
    /// resolves names through [`crate::SubRegistry::find_by_name`] in the same step as the rebind,
    /// so a stale id is structurally unexpected: the variant surfaces a routing breach rather than
    /// a benign no-op. Distinct from [`Self::DetachUnknownSub`] (a stale-id detach attempt) and
    /// from [`Self::EffectCompleteForUnknownSub`] (a stray completion arrival).
    RebindUnknownSub { sub: SubId },
    /// A discovery template's reconcile pass matched `path` and minted a dynamic Sub for it; `kind`
    /// is the snapshot's kind for the matched terminus. `source` is the discovery Sub the mint ran
    /// for. Operator narration — the bin logs it as a mint observed.
    DiscoveryMinted {
        source: SubId,
        path: Arc<Path>,
        kind: ResourceKind,
    },
    /// A discovery template's live minted-Sub count (derived from `SubRegistry`) crossed the
    /// warning threshold for the first time — the pattern is matching more targets than typical
    /// (likely too broad, e.g. `/*` without further constraint). One-shot per template lifetime;
    /// the latch on `Sub.fanout_warned` suppresses repeats.
    DiscoveryFanoutThreshold { source: SubId, count: usize },
    /// A dynamic Sub minted by the discovery template `source` at `path` was reaped because its
    /// anchor disappeared. Operator narration; if the path re-materialises the next reconcile
    /// re-mints it (under a fresh [`SubId`]). Pairs with the per-Sub [`Self::SubDetached`]
    /// carrying [`DetachReason::AnchorLost`].
    DiscoverySubReaped {
        source: SubId,
        sub: SubId,
        path: Arc<Path>,
    },
    /// A burst-lifecycle helper was invoked on a Profile whose state did not match the helper's
    /// typed precondition (e.g., `transition_to_verifying` called on a Profile in
    /// `ActivePostFire`). The helper bails before mutating any state; this variant surfaces the
    /// routing breach so the originating dispatcher can be fixed.
    ///
    /// `helper` names which entry point bailed; `observed` reports the Profile's state at the call.
    /// Stale `ProfileId` (`profiles.get(_) == None`) does NOT emit this variant — that is a benign
    /// post-detach race, surfaced only when an op targeting the missing slot rises through the
    /// usual handlers.
    ///
    /// Structurally unreachable in v1 — every dispatcher gates on the state variant before reaching
    /// a helper — but the precondition gates the gate, so any future routing regression surfaces
    /// here instead of silently dropping the transition.
    InvalidBurstTransition {
        profile: ProfileId,
        helper: BurstHelper,
        observed: ProfileStateDiscriminant,
    },
    /// A probe response's payload shape contradicts the route the engine requested: a `Verifying` /
    /// `Rebasing` (proof) probe received the structural `DirEnumerated`, or a `Descent` probe
    /// received an `AnchorOk` / `SubtreeProven` proof. The engine recovers route-appropriately — a
    /// burst finishes to `Idle` preserving its anchor/baseline (a walker defect is not an
    /// anchor-identity change), a descent abandons its prefix — and is self-healing (a later
    /// `FsEvent` re-drives the burst / a fresh descent).
    ///
    /// Distinct from siblings on the same response path:
    /// - [`Self::StaleProbeResponse`] — a *correlation* drift; here the correlation matched (the
    ///   response gate proved the response correlates to the live carrier), so this variant carries
    ///   no correlation — it would be pure noise.
    /// - [`Self::AnchorKindMismatch`] — a *kind* divergence on a successfully-lowered snapshot
    ///   (`File` vs `Dir`), not a payload-shape violation. That arm operates one step later, on a
    ///   response that already parsed as a proof.
    ///
    /// Structurally unreachable in v1 — the emission choke kinds each probe off the owner's state
    /// and the pool dispatches 1:1, so a correct walker never returns a shape the route cannot
    /// accept. An emission signals a walker-side routing regression.
    WalkerContractViolated { owner: ProfileId },
}
