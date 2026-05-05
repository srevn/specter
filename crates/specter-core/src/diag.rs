//! Engine diagnostics.
//!
//! Emitted alongside the dropped/clamped Inputs they explain. Variant set
//! grows phase-by-phase as new drop paths land. Each variant is light-weight
//! (a few small fields) and carries enough context to log meaningfully.

use crate::ids::{ProfileId, ResourceId, SubId, TimerId};
use crate::input::FsEvent;
use crate::op::ProbeCorrelation;
use crate::profile::BurstIntent;

/// Engine-emitted diagnostic. Equality is structural so tests can pin the
/// exact variant + fields produced by a given dropped Input.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum Diagnostic {
    /// `ProbeResponse` for a Profile not in `BurstPhase::Probing`, or for a
    /// `Probing` whose `correlation` doesn't match the response.
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
    /// A pending-path descent was abandoned because the kernel rejected
    /// the watch on its prefix (`Input::WatchOpRejected` arrived for the
    /// resource the descent was probing). The clamp atomically zeroed
    /// `watch_demand`, dropping the descent's contribution; the engine
    /// drops the descent state to prevent a downstream
    /// `sub_watch_demand` underflow when the late probe response or a
    /// rewind would otherwise try to release the already-released
    /// contribution. Recovery is operator-driven (re-attach via SIGHUP)
    /// or, if the parent ancestor is itself watched, automatic via the
    /// next reconciliation.
    PendingDescentVacated {
        profile: ProfileId,
        prefix: ResourceId,
        errno: i32,
    },
    /// A path-based attach request carried a malformed `PathBuf` —
    /// empty, containing `.` / `..` (caller should canonicalize), or
    /// carrying a Windows path prefix (unsupported on Unix v1). The
    /// engine drops the attach. Defense-in-depth: config validation is
    /// the canonical guard, but the engine surfaces the reason separately
    /// so a misuse from the bin or hot reload is visible.
    AttachPathInvalid { hint: &'static str },
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
