//! Engine input variants and the normalized `FsEvent`.

use crate::effect::{DedupKey, EffectOutcome};
use crate::ids::{ProfileId, ResourceId, SubId, TimerId};
use crate::op::{ProbeResponse, WatchOp};
use crate::profile::TimerKind;
use crate::sub::SubRegistryDiff;

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

#[non_exhaustive]
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
        errno: i32,
    },
    ConfigDiff(SubRegistryDiff),
}
