//! Engine input variants and the normalized `FsEvent`.

use crate::effect::{DedupKey, EffectOutcome};
use crate::ids::{ProfileId, ResourceId, SubId, TimerId};
use crate::op::{ProbeResponse, WatchFailure, WatchOp};
use crate::profile::TimerKind;
use crate::promoter::{PromoterAttachRequest, PromoterRegistryDiff};
use crate::sub::{SubAttachRequest, SubRegistryDiff};

/// Normalized filesystem event. `kqueue` / `inotify` / `FSEvents` flags
/// fold into these six.
///
/// Identity events ([`Removed`] / [`Renamed`] / [`Revoked`]) are slot-
/// level: they fire on the watched inode itself. Backends emit them via
/// `IN_DELETE_SELF` / `IN_MOVE_SELF` / `IN_UNMOUNT` (inotify) or
/// `NOTE_DELETE` / `NOTE_RENAME` / `NOTE_REVOKE` (kqueue) on the watched
/// resource — they never name a child, even when the kernel could
/// (inotify's `IN_CREATE` etc. carry a basename; v1 throws it away and
/// folds into [`StructureChanged`] so the engine probes the parent for
/// the delta).
///
/// Name-bearing structure events (`IN_CREATE` / `IN_DELETE` /
/// `IN_MOVED_*` on inotify; `NOTE_WRITE` on a kqueue Dir) collapse into
/// [`StructureChanged`] — the engine probes the parent on each such
/// event to discover what changed by name.
///
/// [`Removed`]: Self::Removed
/// [`Renamed`]: Self::Renamed
/// [`Revoked`]: Self::Revoked
/// [`StructureChanged`]: Self::StructureChanged
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum FsEvent {
    #[default]
    Modified,
    MetadataChanged,
    StructureChanged,
    Renamed,
    Removed,
    Revoked,
}

/// Scope of a sensor overflow signal.
///
/// inotify's `IN_Q_OVERFLOW` is queue-wide ([`Global`]); FSEvents emits
/// per-stream overflow ([`Resource`]). The v1 inotify backend always
/// emits `Global`; kqueue never emits overflow under v1 (`EV_CLEAR`
/// coalesces but never silently drops at the kernel level).
///
/// Carried on the sensor → engine path in two places: in the
/// per-`poll_until` drain (`specter-sensor::WatcherEvent::Overflow`)
/// and in the engine input variant the bin lifts it into
/// ([`Input::SensorOverflow`]).
///
/// [`Global`]: Self::Global
/// [`Resource`]: Self::Resource
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum OverflowScope {
    /// Per-watched-resource scope. FSEvents reports overflow per active
    /// stream; the v1 inotify backend never emits this variant.
    Resource(ResourceId),
    /// Watcher-wide queue overflow; affects every active watch on the
    /// sensor. inotify's `IN_Q_OVERFLOW` is the only emitter under v1.
    Global,
}

#[derive(Debug, Clone)]
pub enum Input {
    FsEvent {
        resource: ResourceId,
        event: FsEvent,
    },
    ProbeResponse(ProbeResponse),
    /// Engine timer fired. `profile` and `kind` are stamped at schedule
    /// time and routed back unchanged; `id` is the lazy-invalidation
    /// epoch that disambiguates a live timer from a superseded one for
    /// the same `(profile, kind)` slot.
    TimerExpired {
        profile: ProfileId,
        kind: TimerKind,
        id: TimerId,
    },
    EffectComplete {
        sub: SubId,
        key: DedupKey,
        result: EffectOutcome,
    },
    WatchOpRejected {
        resource: ResourceId,
        op: WatchOp,
        failure: WatchFailure,
    },
    ConfigDiff(WatchRegistryDiff),
    /// Sensor reports it dropped events at the kernel level — the watch
    /// state is intact but the event stream is no longer trustworthy
    /// over `scope`. The engine response is to reseed every Profile in
    /// scope (`Engine::on_sensor_overflow`): cancel any in-flight burst
    /// and start a fresh Seed burst whose post-probe `dispatch_seed_ok`
    /// re-establishes baseline against disk reality and runs the
    /// drift detection. Active-mode drift (overflow path: `baseline`
    /// persists across the reseed) compares `baseline.hash()` against
    /// the post-graft `current.hash()`; survival-mode drift (anchor-
    /// loss recovery path) compares `last_settled_hash_at_loss` against
    /// `current.hash()`. On drift, fires once for every Subtree-scoped
    /// key in `fired_subs`, then rebases.
    ///
    /// Always [`OverflowScope::Global`] on the v1 inotify backend
    /// (`IN_Q_OVERFLOW` is queue-wide). The [`OverflowScope::Resource`]
    /// variant exists for FSEvents (per-stream overflow) and to keep
    /// the engine's handler shape stable across backends.
    SensorOverflow {
        scope: OverflowScope,
    },
    /// Attach a Sub. The engine resolves the request's anchor (path
    /// or resource), mints a fresh Profile if `(anchor, config_hash)`
    /// doesn't already index one, registers the Sub, and starts the
    /// Seed burst (immediate path) or descent (pending path).
    ///
    /// The minted [`SubId`] surfaces via [`crate::Diagnostic::SubAttached`] in the
    /// `StepOutput.diagnostics` stream; on path rejection
    /// (`Tree::parse_attach_path` failure), surfaces via
    /// [`crate::Diagnostic::AttachPathInvalid`] instead with no Sub registered.
    /// The bin's loader keys its `name → SubId` map off the diagnostic
    /// stream rather than the (deleted) synchronous-return shape.
    AttachSub(SubAttachRequest),
    /// Detach a Sub. The engine drops the Sub from the registry,
    /// decrements `Profile.sub_refcount`, and either reaps the
    /// Profile (Idle/Pending: immediate) or marks it for
    /// deferred-reap (Active: [`crate::BurstFinish::Reap`]). Stale
    /// [`SubId`] yields a [`crate::Diagnostic::DetachUnknownSub`].
    DetachSub(SubId),
    /// Attach a Promoter. The engine renders the literal-prefix path,
    /// materialises the Tree, opens the probe channel, and starts the
    /// Promoter in either `Active` (prefix materialised) or
    /// `PrefixPending` (descent needed). The minted
    /// [`crate::PromoterId`] surfaces via
    /// [`crate::Diagnostic::PromoterAttached`]; on
    /// path rejection, via [`crate::Diagnostic::AttachPathInvalid`].
    AttachPromoter(PromoterAttachRequest),
}

/// Hot-reload diff payload for [`Input::ConfigDiff`] — the watch-registry
/// generalisation that carries both Sub and Promoter changes.
///
/// Composes the Sub side ([`SubRegistryDiff`]) and the Promoter side
/// ([`PromoterRegistryDiff`]). The engine's `on_config_diff` applies
/// both halves atomically in one step: Sub removals → Sub modifications
/// → Sub additions → Promoter removals → Promoter modifications →
/// Promoter additions, all merging into a single sorted
/// [`crate::StepOutput`].
///
/// `Default` is derived so call sites that touch only one half can
/// construct via struct-update syntax: `WatchRegistryDiff { subs,
/// ..Default::default() }`.
#[derive(Clone, Debug, Default)]
pub struct WatchRegistryDiff {
    pub subs: SubRegistryDiff,
    pub promoters: PromoterRegistryDiff,
}
