//! Watch and Probe ops, plus their request/response payloads.

use crate::ids::{ProfileId, ResourceId};
use crate::resource::ResourceKind;
use crate::scan_config::ScanConfig;
use crate::snapshot::tree::{DirSnapshot, TreeSnapshot};
use crate::sub::ClassSet;
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

/// Engine-monotonic correlation token — pairs each `ProbeRequest` with the
/// `ProbeResponse` that answers it.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct ProbeCorrelation(pub u64);

#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ProbeKind {
    #[default]
    File,
    Directory,
}

/// Stateless probe request.
///
/// The Sensor's Prober pool is stateless re Profile — every field needed
/// to perform the syscall lives on the message. `scan_config` is cloned at
/// emission time; `correlation` pairs the response with the engine-side
/// `BurstPhase::Verifying` slot. Cloning is allocation-cheap on the hot path:
/// `baseline_subtree` is `Arc::clone` and `force_walk` is a small
/// `BTreeSet`.
#[derive(Clone, Debug)]
pub struct ProbeRequest {
    /// Profile this probe belongs to. The Sensor uses it for the
    /// expectation map and echoes it back on `ProbeResponse`.
    pub profile: ProfileId,
    /// Engine-monotonic correlation token — pairs request with response.
    pub correlation: ProbeCorrelation,
    /// File vs Directory dispatch. Walker dispatches on this; an on-disk
    /// kind mismatch returns `Vanished`.
    pub kind: ProbeKind,
    /// Resource the prober walks. Often `Profile.resource` (Seed bursts);
    /// for Standard bursts this becomes the LCA of dirty resources. The
    /// walker stamps it onto `DirSnapshot.root_resource` (advisory) but
    /// otherwise doesn't consult it — the walker has no Tree.
    pub target_resource: ResourceId,
    /// Filesystem path of `target_resource` at probe-emission time.
    /// Engine builds via `tree.path_of(target_resource)`. Empty `PathBuf`
    /// is the lone failure mode; the walker treats empty as `Vanished`.
    pub target_path: PathBuf,
    /// `ScanConfig` to honour (recursive, hidden, exclude, pattern,
    /// `max_depth`). Cloned at emit time.
    pub scan_config: ScanConfig,
    /// `Profile.config_hash` at emission time. Walker stamps every
    /// `DirSnapshot.captured_with` so two Profiles sharing a Resource
    /// with different filters cannot produce identical `dir_hash` for
    /// divergent in-scope content.
    pub captured_with: u64,
    /// Engine's last-known view of `target_resource`'s subtree. The
    /// walker consults `baseline_subtree.root_meta` for mtime-skip and
    /// propagates child baselines via `entries[name].subtree`. `None`
    /// means "no prior observation": first Seed of a fresh Profile.
    ///
    /// Cheap to ship — `Arc::clone` on the channel send. Multiple
    /// workers may hold the same Arc concurrently (immutable
    /// post-construction except for the `OnceLock<u128>` hash cache,
    /// which is `Sync`).
    pub baseline_subtree: Option<Arc<DirSnapshot>>,
    /// Set of paths the walker MUST enumerate (refusing mtime-skip) at
    /// any directory whose path equals one of these OR is an ancestor of
    /// one. Populated from `dirty_resources ∩ subtree(target_resource)`:
    /// kqueue events that arrived since the last probe at this target.
    ///
    /// Walker checks `force_walk.iter().any(|p| p.starts_with(current))`
    /// — O(N) per directory, N = `|force_walk|` (typically 1–5). The set
    /// is *minimal* (only the dirty paths); the walker's prefix-match
    /// covers the "ancestor of forced descendant" case without engine-side
    /// closure construction.
    ///
    /// `BTreeSet<PathBuf>` (not `Vec<PathBuf>`) so iteration order is
    /// deterministic for replay.
    pub force_walk: BTreeSet<PathBuf>,
    /// `true` ⇒ walker bypasses mtime-skip at every directory regardless
    /// of `baseline_subtree` and `force_walk`. Engine sets this when
    /// `Burst.forced` is true (max-settle deadline elapsed; force-fire).
    ///
    /// Defensive: mtime-skip is correct under normal semantics, but a
    /// forced probe wants the freshest possible snapshot regardless of cost.
    pub forced: bool,
}

#[derive(Debug, Clone)]
pub enum ProbeResult {
    /// Successful probe: a `TreeSnapshot` rooted at `target_resource`.
    /// The variant of the inner `TreeSnapshot` matches `req.kind`:
    /// File ⇒ `TreeSnapshot::File(LeafEntry)`; Directory ⇒
    /// `TreeSnapshot::Dir(Arc<DirSnapshot>)`. A walker-internal kind
    /// mismatch on the on-disk entry yields `Vanished`, never a
    /// kind-mismatched `Ok`.
    Ok(TreeSnapshot),
    /// Path doesn't exist (`ENOENT`) or kind doesn't match `req.kind`
    /// (file probe found dir, dir probe found file). The engine routes
    /// `Vanished` through `on_anchor_terminal_event`.
    Vanished,
    /// Other I/O error at the *root* of the probe (root `lstat`,
    /// permission, etc.). Mid-walk errors don't produce `Failed` — they
    /// skip-and-continue with `tracing::warn!`.
    Failed { errno: i32 },
}

#[derive(Debug, Clone)]
pub struct ProbeResponse {
    pub profile: ProfileId,
    pub correlation: ProbeCorrelation,
    pub result: ProbeResult,
}

#[derive(Debug, Clone)]
pub enum WatchOp {
    /// Install (or re-register) a watch on `resource` at `path`.
    ///
    /// `kind` is the engine's authoritative classification of the slot
    /// (`File` / `Dir` / `Unknown`). The sensor uses it as a verification
    /// step against the inode its `O_PATH` / `open` fd resolved to —
    /// rejecting installs where the path's on-disk kind diverges from
    /// the engine's expectation. `Unknown` is a wildcard: the engine
    /// emits it for slots it has not yet classified (descent prefix
    /// placeholder, post-`add_watch_demand` before the first probe), and
    /// the sensor accepts whatever inode resolves while caching the
    /// observed kind for normalization / mask translation.
    ///
    /// `events` is the L3 carrier for the per-Resource event-class union
    /// (R2 / D4): the engine ships `Resource.events_union` on every
    /// `Watch` op, the sensor diffs the cached per-FD mask, and
    /// re-registers iff different. `ClassSet::EMPTY` degrades to
    /// identity-floor-only delivery (kqueue: `NOTE_DELETE | NOTE_RENAME
    /// | NOTE_REVOKE`).
    Watch {
        resource: ResourceId,
        path: PathBuf,
        kind: ResourceKind,
        events: ClassSet,
    },
    Unwatch {
        resource: ResourceId,
    },
    Suppress {
        resource: ResourceId,
    },
    Unsuppress {
        resource: ResourceId,
    },
}

impl Default for WatchOp {
    /// Sentinel placeholder so `WatchOp` satisfies `tinyvec::Array`'s
    /// `T: Default` bound. Inline `TinyVec` slots are overwritten before
    /// they are ever read.
    fn default() -> Self {
        Self::Unsuppress {
            resource: ResourceId::default(),
        }
    }
}

/// Typed failure of a [`WatchOp::Watch`] install.
///
/// The engine demuxes on the variant — never on `errno` — so backends can
/// map their kernel-specific errno values once at the trait boundary
/// without forcing the engine to learn each kernel's vocabulary. Each
/// variant names *what the engine should do*; the inner `errno` is
/// diagnostic context, not a behavioural switch.
///
/// The `From<io::Error>` translation lives in `specter-sensor` (via
/// `WatchFailureExt::from_io`) because errno-name matching needs `libc`,
/// which is banned in `core` per `deny.toml`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum WatchFailure {
    /// Backpressure: kernel resource limit reached. Engine clamps
    /// `watch_demand := 0` on the affected resource and waits for natural
    /// retry on the parent's next reconcile.
    ///
    /// kqueue: `EMFILE` / `ENFILE` from `open(O_EVTONLY)`.
    /// inotify: `ENOSPC` from `inotify_add_watch` (`max_user_watches`).
    Pressure { errno: i32 },

    /// Path-fatal: the path doesn't resolve to an inode the engine
    /// expects. Engine treats as terminal for the resource and re-resolves
    /// via descent (anchor case ⇒ `finalize_anchor_lost`; descendant case
    /// ⇒ clamp + wait for parent).
    ///
    /// kqueue: `ENOENT` / `EACCES` / `ELOOP` / `ENOTDIR`.
    /// inotify: same set, plus `ENOTDIR` under `IN_ONLYDIR`.
    Resource { errno: i32 },

    /// Programmer error or trait-misuse: the watcher's invariant has been
    /// violated. Engine logs at error level and clamps the slot; in
    /// practice these never fire on a healthy bin. Examples: path with
    /// embedded NUL, `EBADF` against the inotify_fd, double-mapping of
    /// one wd to two `ResourceId`s (hardlink aliasing).
    Invariant { errno: i32 },
}

impl WatchFailure {
    /// Underlying errno carried by every variant. Convenience for
    /// diagnostic logging — equivalent to a three-arm `match`.
    #[must_use]
    pub const fn errno(&self) -> i32 {
        match self {
            Self::Pressure { errno } | Self::Resource { errno } | Self::Invariant { errno } => {
                *errno
            }
        }
    }
}

// `ProbeRequest` carries baseline_subtree/force_walk/forced etc.
// `Probe` is the dominant variant — every burst emits one — and boxing
// it would add an allocation per probe with no observable benefit since
// `Cancel` is sparse (per Profile reap, not per burst). The size delta
// rides on `tinyvec` inline slots, which is why we accept it explicitly.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ProbeOp {
    Probe { request: ProbeRequest },
    Cancel { profile: ProfileId },
}

impl Default for ProbeOp {
    /// Sentinel for `tinyvec::Array`. The `Cancel` variant carries no
    /// `ProbeRequest`, sidestepping a `Default` requirement on that type.
    fn default() -> Self {
        Self::Cancel {
            profile: ProfileId::default(),
        }
    }
}
