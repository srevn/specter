//! Engine diagnostics.
//!
//! Emitted alongside the dropped/clamped Inputs they explain. Variant set
//! grows phase-by-phase as new drop paths land. Each variant is light-weight
//! (a few small fields) and carries enough context to log meaningfully.

use crate::ids::{ProfileId, PromoterId, ResourceId, SubId, TimerId};
use crate::input::{FsEvent, OverflowScope};
use crate::op::{ProbeCorrelation, ProbeOwner, WatchFailure};
use crate::profile::BurstIntent;
use crate::resource::ResourceKind;
use compact_str::CompactString;
use std::path::PathBuf;

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

/// Engine-emitted diagnostic. Equality is structural so tests can pin the
/// exact variant + fields produced by a given dropped Input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Diagnostic {
    /// `ProbeResponse` whose `(owner, correlation)` doesn't match the
    /// owner's live probe channel. Catches stale-id (post-detach),
    /// post-cancel arrivals, and out-of-order responses across all
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
    /// [`crate::BurstPhase::Awaiting`]. Two paths reach here legitimately:
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
    /// `Engine::detach_sub` (or `Input::ConfigDiff::removed`) targeted a
    /// `SubId` not in the registry. Reachable when hot reload races with
    /// a previous detach, or when an external caller submits a stale id.
    /// Distinct from [`Self::EffectCompleteForUnknownSub`] — that variant
    /// fires on a stray completion arrival, not on a detach miss.
    DetachUnknownSub { sub: SubId },
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
    /// `Profile.reap_pending` was set, then a fresh `attach_sub` arrived
    /// at the same `(resource, config_hash)` before the burst completed —
    /// the deferred reap is cancelled and the Profile remains alive under
    /// the new Sub set. Informational — not an error. Pairs with
    /// [`Self::ReapPendingResolved`] (the reap actually ran).
    ReapPendingCancelled { profile: ProfileId },
    /// `Profile.reap_pending` was set; the burst completed and the Profile
    /// has been reaped. Informational — not an error.
    ReapPendingResolved { profile: ProfileId },
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
    /// A path-based attach request carried a malformed `PathBuf` —
    /// empty, containing `.` / `..` (caller should canonicalize), or
    /// carrying a Windows path prefix (unsupported on Unix v1). The
    /// engine drops the attach. Defense-in-depth: config validation is
    /// the canonical guard, but the engine surfaces the reason separately
    /// so a misuse from the bin or hot reload is visible.
    ///
    /// `path` carries the offending request so operators submitting
    /// multi-path attach batches (hot reload `ConfigDiff::added`) can
    /// identify which entry failed without re-scanning the config.
    AttachPathInvalid { path: PathBuf, hint: &'static str },
    /// A descent dispatch ran with `DescentState.remaining_components`
    /// empty. The invariant on `DescentState` (see `core/profile.rs`)
    /// says this can't happen: the anchor itself is the last remaining
    /// component, and descent transitions Pending → Idle on
    /// materialization rather than emptying the vec. If it ever fires,
    /// it's a state-machine bug — the diagnostic surfaces the breach
    /// and the engine takes the conservative recovery path
    /// (`release_descent_prefix_claim`, returning the Profile to Idle
    /// without leaking the prefix's `watch_demand` contribution).
    DescentInvariantViolation {
        profile: ProfileId,
        prefix: ResourceId,
    },
    /// `splice` could not navigate from the prior snapshot's anchor down
    /// to `target`. Two structural causes:
    /// - `target` is outside the anchor's tree subtree (e.g., stale
    ///   `ResourceId`, or a scope contraction that revoked coverage of
    ///   the probed path).
    /// - The path crossed a `subtree: None` intermediate (snapshot
    ///   coverage gap — the walker stored the entry but did not recurse).
    ///
    /// Engine contract is "graft only into observed subtrees", so this
    /// path indicates a contract violation. The variant exists to
    /// surface the breach in operator logs; the engine falls back to
    /// keeping its prior view (no integration of `replacement`) and
    /// converges on the next probe.
    ///
    /// File-anchored Profiles never call `splice` (their Profile.current
    /// is `TreeSnapshot::File(leaf)`, integrated by an inline write at
    /// dispatch time, never grafted) — so only the Dir-prior structural
    /// triggers above remain. Structurally unreachable in v1;
    /// defense-in-depth against future scope changes.
    SpliceCrossedUncovered {
        profile: ProfileId,
        target: ResourceId,
    },
    /// `FsEvent` arrived while the Profile was in
    /// [`crate::BurstPhase::Awaiting`] or [`crate::BurstPhase::Rebasing`]
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
    /// burst to [`crate::BurstPhase::Rebasing`] so it can re-establish
    /// a baseline against disk reality. Late completions land in
    /// [`Self::EffectCompleteOutsideAwaiting`].
    AwaitGateDeadlineElapsed {
        profile: ProfileId,
        outstanding: u32,
    },
    /// `Input::SensorOverflow` arrived from the Sensor — the kernel's
    /// event queue dropped record(s) over `scope` and the watcher
    /// surfaced the loss-of-trust signal. The engine reseeded every
    /// in-scope Profile via `start_seed_burst`; the diagnostic surfaces
    /// the event in operator logs so the underlying load condition
    /// (`max_queued_events` saturation, slow downstream actuator
    /// blocking the watcher's drain) can be tuned.
    ///
    /// One emission per overflow record. The emitted variant is the
    /// engine's only signal that "we missed events" — the bursts the
    /// reseed schedules carry no per-Profile annotation that they were
    /// triggered by overflow rather than a normal `FsEvent`.
    SensorOverflow { scope: OverflowScope },
    /// A Promoter has been registered with the engine and assigned
    /// `promoter`. Emitted by `attach_promoter`. The bin layer reads
    /// the variant from `StepOutput.diagnostics` to reconcile its
    /// `name → PromoterId` mapping after a hot-reload diff applies.
    /// `name` carries the Promoter's user-facing name verbatim so the
    /// reload path can update its lookup table without re-reading the
    /// engine's registry.
    PromoterAttached {
        promoter: PromoterId,
        name: CompactString,
    },
    /// A Promoter has been removed from the engine. Pairs with
    /// [`Self::PromoterAttached`]; the bin removes the entry from its
    /// `name → PromoterId` map on receipt.
    PromoterReaped { promoter: PromoterId },
    /// A Promoter's literal-prefix descent ran with
    /// `DescentState.remaining_components` empty — analogue of
    /// [`Self::DescentInvariantViolation`] for the Promoter side.
    /// Should be unreachable per the invariant on `DescentState`; the
    /// engine surfaces the breach + retains state without leaking the
    /// prefix watch.
    PromoterDescentInvariantViolation {
        promoter: PromoterId,
        prefix: ResourceId,
    },
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
    /// Promoter enumeration matched `path` and the engine has minted a
    /// dynamic Sub for it. `kind` is the kind the snapshot reports for
    /// the matched entry. The bin uses this for operator-visible
    /// "promotion observed" logs; the engine's own bookkeeping is in
    /// `Promoter.dynamic_subs`.
    PromotionKindObserved {
        promoter: PromoterId,
        path: PathBuf,
        kind: ResourceKind,
    },
    /// Promoter's `dynamic_subs.len()` crossed a threshold for the
    /// first time. Operator signal that the pattern is matching more
    /// targets than typical — likely a too-broad pattern (e.g. `/*`
    /// without further constraint). One-shot per Promoter lifetime;
    /// the latch on `Promoter.warned_at_threshold` suppresses repeats.
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
}
