//! Watch and Probe ops, plus their request/response payloads.

use crate::ids::{ProfileId, ResourceId};
use crate::scan_config::ScanConfig;
use crate::snapshot::tree::{DirSnapshot, TreeSnapshot};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;

/// Per-watch hints carried alongside `WatchOp::Watch`.
///
/// Engine emission sites pass `WatchOpts::default()` in v1; both fields are
/// reserved for v2 backends that can honor them (`inotify`, `FSEvents`,
/// `ReadDirectoryChangesW`). The kqueue Watcher ignores both fields: kqueue
/// does not support kernel-side recursive watches — recursion is
/// engine-driven via reconciliation — and `O_NOFOLLOW` is unconditionally
/// applied. The fields exist so v2 can opt in without a trait-shape break.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct WatchOpts {
    pub follow_symlinks: bool,
    pub recursive: bool,
}

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
/// `BurstPhase::Probing` slot. Cloning is allocation-cheap on the hot path:
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
    Watch {
        resource: ResourceId,
        path: PathBuf,
        opts: WatchOpts,
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
