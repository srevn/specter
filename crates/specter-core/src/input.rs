//! Engine input variants and the normalized `FsEvent`.

use crate::effect::{DedupKey, EffectOutcome};
use crate::ids::{ProfileId, ResourceId, SubId, TimerId};
use crate::op::{ProbeResponse, WatchOp};
use crate::profile::TimerKind;
use crate::sub::SubRegistryDiff;

/// Normalized filesystem event. `kqueue` / `inotify` / `FSEvents` flags fold
/// into these six.
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
