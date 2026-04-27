//! Engine input variants and the normalized `FsEvent`.

use crate::effect::{DedupKey, EffectOutcome};
use crate::ids::{ResourceId, SubId, TimerId};
use crate::op::{ProbeResponse, WatchOp};
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
    TimerExpired(TimerId),
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
