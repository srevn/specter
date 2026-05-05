//! Engine diagnostics.
//!
//! Emitted alongside the dropped/clamped Inputs they explain. Variant set
//! grows phase-by-phase as new drop paths land. Each variant is light-weight
//! (a few small fields) and carries enough context to log meaningfully.

use crate::ids::{ProfileId, ResourceId, SubId, TimerId};
use crate::input::FsEvent;
use crate::op::ProbeCorrelation;
use crate::profile::BurstIntent;

/// Which Profile-side claim was the subject of a [`Diagnostic::ProfileClaimPurged`]
/// emission. Each claim type has a dedicated bookkeeping field on
/// [`crate::profile::Profile`]:
/// - [`Self::Anchor`] ⇔ `Profile.anchor_contribution = true`
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
    /// `ProbeResponse` for a Profile not in `BurstPhase::Verifying`, or for a
    /// `Verifying` whose `correlation` doesn't match the response.
    StaleProbeResponse {
        profile: ProfileId,
        correlation: ProbeCorrelation,
    },
    /// `TimerExpired(id)` whose `TimerId` is not referenced by any Profile's
    /// burst. `pop_expired` already drops these silently; the variant is the
    /// defense-in-depth signal for a direct `step(Input::TimerExpired)` call
    /// from a misbehaving caller.
    StaleTimer { id: TimerId },
    /// `EffectComplete::Ok` arriving while the Profile is `Active`. The
    /// active burst's eventual Effect's own `EffectComplete::Ok` will fire
    /// the next Seed.
    EffectCompleteWhileActive { sub: SubId, profile: ProfileId },
    /// `EffectComplete` for a Sub not in the registry.
    EffectCompleteForUnknownSub { sub: SubId },
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
    /// L5 `fs_event_to_class` mapping) is not in the covering Profile's
    /// `events_union`. The user opted out of this class via `Sub.events`,
    /// so the engine drops the event before it can drive a burst (per
    /// design §6.1 — class filter sits before dirty-set bumps).
    ///
    /// Distinct from [`Self::EventNoConsumer`] (no covering Profile at
    /// all): there *is* a covering Profile, but the class filter rejects.
    /// Distinct from the prior v1 `MetadataChangedIgnored`: that variant
    /// hard-coded "always drop METADATA"; this one carries the dropped
    /// event + Profile so logs can disambiguate user opt-out from race.
    ///
    /// Anchor-on-Profile events bypass this filter unconditionally per
    /// design D8 — lifecycle continuity (anchor terminal events drive
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
    WatchOpRejected { resource: ResourceId, errno: i32 },
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
        errno: i32,
    },
    /// A path-based attach request carried a malformed `PathBuf` —
    /// empty, containing `.` / `..` (caller should canonicalize), or
    /// carrying a Windows path prefix (unsupported on Unix v1). The
    /// engine drops the attach. Defense-in-depth: config validation is
    /// the canonical guard, but the engine surfaces the reason separately
    /// so a misuse from the bin or hot reload is visible.
    AttachPathInvalid { hint: &'static str },
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
    /// Structurally unreachable in v1; the variant is defense-in-depth
    /// against future scope changes that might reach it.
    SpliceCrossedUncovered {
        profile: ProfileId,
        target: ResourceId,
    },
}

impl Default for Diagnostic {
    /// Sentinel for `tinyvec::Array`'s `T: Default` bound on
    /// `StepOutput.diagnostics`. Inline slots are overwritten before they
    /// are ever read.
    fn default() -> Self {
        Self::StaleTimer {
            id: TimerId::default(),
        }
    }
}
