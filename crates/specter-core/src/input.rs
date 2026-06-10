//! Engine input variants and the normalized `FsEvent`.

use std::time::Duration;

use crate::effect::EffectCompletion;
use crate::ids::{ProfileId, ResourceId, SubId, TimerId};
use crate::op::{ProbeResponse, WatchFailure};
use crate::profile::TimerKind;
use crate::sub::{SubAttachRequest, SubRegistryDiff};

/// Normalized filesystem event. `kqueue` / `inotify` / `FSEvents` flags fold into these six.
///
/// Identity events ([`Removed`] / [`Renamed`] / [`Revoked`]) are slot- level: they fire on the
/// watched inode itself. Backends emit them via `IN_DELETE_SELF` / `IN_MOVE_SELF` / `IN_UNMOUNT`
/// (inotify) or `NOTE_DELETE` / `NOTE_RENAME` / `NOTE_REVOKE` (kqueue) on the watched resource — they
/// never name a child, even when the kernel could (inotify's `IN_CREATE` etc. carry a basename; v1
/// throws it away and folds into [`StructureChanged`] so the engine probes the parent for the delta).
///
/// Name-bearing structure events (`IN_CREATE` / `IN_DELETE` / `IN_MOVED_*` on inotify; `NOTE_WRITE`
/// on a kqueue Dir) collapse into [`StructureChanged`] — the engine probes the parent on each such
/// event to discover what changed by name.
///
/// [`Removed`]: Self::Removed [`Renamed`]: Self::Renamed [`Revoked`]: Self::Revoked
/// [`StructureChanged`]: Self::StructureChanged
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum FsEvent {
    #[default]
    ContentChanged,
    MetadataChanged,
    StructureChanged,
    Renamed,
    Removed,
    Revoked,
}

impl FsEvent {
    /// Identity (slot-level) events — `Removed` / `Renamed` / `Revoked`. They fire on the watched
    /// inode itself, are terminal when they reach the anchor (`on_anchor_terminal_event`), and
    /// reconcile a covered descendant through the diff-against-prior pass. Each is a distinct
    /// lifecycle fact routed structurally — never a recency hint, so never eligible for same-tick
    /// coalescing.
    #[must_use]
    pub const fn is_identity(self) -> bool {
        matches!(self, Self::Removed | Self::Renamed | Self::Revoked)
    }

    /// Recency-class events — `ContentChanged` / `MetadataChanged` / `StructureChanged`. Each is a
    /// lossy "this resource changed in this class" hint whose sole truth is the next probe; the
    /// exact complement of [`Self::is_identity`].
    #[must_use]
    pub const fn is_recency(self) -> bool {
        !self.is_identity()
    }
}

/// Scope of a sensor overflow signal.
///
/// inotify's `IN_Q_OVERFLOW` is queue-wide ([`Global`]); FSEvents emits per-stream overflow
/// ([`Resource`]). The v1 inotify backend always emits `Global`; kqueue never emits overflow under
/// v1 (`EV_CLEAR` coalesces but never silently drops at the kernel level).
///
/// Carried on the sensor → engine path in two places: in the per-`drain_ready` drain
/// (`specter-sensor::WatcherEvent::Overflow`) and in the engine input variant the bin lifts it into
/// ([`Input::SensorOverflow`]).
///
/// [`Global`]: Self::Global [`Resource`]: Self::Resource
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OverflowScope {
    /// Per-watched-resource scope. FSEvents reports overflow per active stream; the v1 inotify
    /// backend never emits this variant.
    Resource(ResourceId),
    /// Watcher-wide queue overflow; affects every active watch on the sensor. inotify's
    /// `IN_Q_OVERFLOW` is the only emitter under v1.
    Global,
}

#[derive(Debug, Clone)]
pub enum Input {
    FsEvent {
        resource: ResourceId,
        event: FsEvent,
    },
    ProbeResponse(ProbeResponse),
    /// Engine timer fired. `profile` and `kind` are stamped at schedule time and routed back
    /// unchanged; `id` is the lazy-invalidation epoch that disambiguates a live timer from a
    /// superseded one for the same `(profile, kind)` slot.
    TimerExpired {
        profile: ProfileId,
        kind: TimerKind,
        id: TimerId,
    },
    /// One effect-completion envelope reaching the engine. Built once at the actuator's wait thread,
    /// threaded unchanged through the controller, lifted into this variant by the bin's wake-bearing
    /// adapter. Tuple-variant shape mirrors [`Input::ProbeResponse`] — both are envelope-bearing
    /// inbound facts, both destructure once at [`crate::Input`]'s engine-side dispatcher.
    EffectComplete(EffectCompletion),
    WatchOpRejected {
        resource: ResourceId,
        failure: WatchFailure,
    },
    /// Hot-reload diff payload. **Name-keyed**: the loader carries operator names, never engine
    /// ids. The engine's `on_config_diff` resolves each name to its live id through the registry's
    /// `by_name` index, then applies the buckets atomically in one step — removals → modifications
    /// → additions — all merging into a single sorted [`crate::StepOutput`].
    ConfigDiff(SubRegistryDiff),
    /// Sensor reports it dropped events at the kernel level — the watch state is intact but the
    /// event stream is no longer trustworthy over `scope`. The engine response is to reseed every
    /// Profile in scope (`Engine::on_sensor_overflow`): cancel any in-flight burst and start a
    /// fresh Seed burst whose post-probe Seed-Ok (`dispatch_quiescence_ok`) re-establishes baseline
    /// against disk reality and runs the drift detection. Active-mode drift (overflow path:
    /// `baseline` persists across the reseed) compares `baseline.hash()` against the post-graft
    /// `current.hash()`; survival-mode drift (anchor-loss recovery path) compares
    /// `last_settled_hash_at_loss` against `current.hash()`. On drift, fires once for every
    /// SubtreeRoot Sub on the Profile that has fired ([`crate::Sub::has_fired`]), then rebases.
    ///
    /// Always [`OverflowScope::Global`] on the v1 inotify backend (`IN_Q_OVERFLOW` is queue-wide).
    /// The [`OverflowScope::Resource`] variant exists for FSEvents (per-stream overflow) and to
    /// keep the engine's handler shape stable across backends.
    SensorOverflow {
        scope: OverflowScope,
    },
    /// Attach a Sub. The engine resolves the request's anchor (path or resource), mints a fresh
    /// Profile if `(anchor, config_hash)` doesn't already index one, registers the Sub, and starts
    /// the Seed burst (immediate path) or descent (pending path).
    ///
    /// The minted [`SubId`] is owned by the engine's registry and resolved by name through its
    /// `by_name` index. A successful attach narrates [`crate::Diagnostic::SubAttached`]; a path
    /// rejection (`Tree::parse_attach_path` failure) narrates
    /// [`crate::Diagnostic::AttachPathInvalid`] with no Sub registered.
    AttachSub(SubAttachRequest),
    /// Detach a Sub. The engine drops the Sub from the registry and either reaps the Profile
    /// (Idle/Pending: immediate, once no Subs remain) or marks it for deferred-reap (Active:
    /// [`crate::BurstFinish::Reap`]). Stale [`SubId`] yields a
    /// [`crate::Diagnostic::DetachUnknownSub`].
    DetachSub(SubId),
    /// Arm the operator `absorb` window on a Profile — the runtime fold-without-fire signal. The
    /// next fireable burst (or the in-flight one, retro-latched) advances the baseline silently
    /// instead of firing, folding an expected replication into the settled reference rather than
    /// echoing it ([`crate::Diagnostic::AbsorbArmed`] on the arm,
    /// [`crate::Diagnostic::QuiescenceAbsorbed`] on each fold).
    ///
    /// `profile` is the honest identity: a window is per-Profile by construction (every Sub on a
    /// Profile folds together), so the driver resolves the operator's Sub name to its Profile
    /// before lifting this — minimal-input discipline shared with [`Self::DetachSub`]. `duration`
    /// is the window length: `None` ⇒ the engine's default (one `settle` interval, consume-on-first
    /// — a one-shot cover for a single replication); `Some(d)` ⇒ a time-boxed window persisting `d`
    /// (covering a run of them). The engine derives `(expiry, mode)` from `duration` + the
    /// Profile's `settle` in one place, so the wire carries only the raw duration.
    ArmAbsorb {
        profile: ProfileId,
        duration: Option<Duration>,
    },
}
