//! Watch and Probe ops, plus their request/response payloads.

use crate::ids::{ProbeCorrelation, ProfileId, PromoterId, ResourceId};
use crate::resource::ResourceKind;
use crate::scan_config::ScanConfig;
use crate::snapshot::tree::{DirSnapshot, LeafEntry};
use crate::sub::ClassSet;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

/// The engine-resident entity that minted a probe — the key the engine
/// demuxes a response back to.
///
/// Echoed verbatim through [`ProbeRequest`] / [`ProbeResponse`] /
/// [`ProbeOp::Cancel`] so the engine can route each response to the
/// entity that's awaiting it.
///
/// Two owner kinds. [`Self::Profile`] drives the burst / descent /
/// rebase lifecycle. [`Self::Promoter`] drives the literal-prefix
/// descent and proxy-enumeration lifecycle. There is no
/// outstanding-probe map this enum keys: "at most one in-flight probe
/// per owner" (I5) is a *representability* property of the owner's
/// single state-resident `ProbeSlot`, so one Profile and one Promoter
/// can each carry an in-flight probe simultaneously without collision
/// by construction.
///
/// **Determinism.** Derived `Ord` (variant order Profile < Promoter,
/// then payload [`ProfileId`] / [`PromoterId`]) is the
/// [`crate::StepOutput::probe_ops`] map key — per-owner
/// last-writer-wins, mirroring the sensor's `expected` map.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Ord, PartialOrd)]
pub enum ProbeOwner {
    /// Profile-driven probe. The engine homes this owner's in-flight
    /// `ProbeCorrelation` on a state-resident `ProbeSlot` (descent /
    /// verify / rebase, one per state variant); the response routes by
    /// inspecting that state.
    Profile(ProfileId),
    /// Promoter-driven probe. The engine homes this owner's in-flight
    /// `ProbeCorrelation` on a state-resident `ProbeSlot` (descent, or
    /// the `Active` enumeration slot tagged with the proxy target);
    /// the response routes by inspecting that state.
    Promoter(PromoterId),
}

/// Engine→walker probe request.
///
/// The variant is the contract: the walker dispatches on it; the engine
/// reads it back at no point. Each variant carries exactly the fields its
/// walker arm consumes — no over-fetching.
///
/// Boxing the heavy `Subtree` variant was considered and rejected: every
/// non-Descent burst produces one, at most one Probe is in flight per
/// burst, and `Arc<DirSnapshot>` baselines are already the dominant
/// payload (the variant width is amortised regardless of how the
/// enclosing `ProbeOp` is stored).
/// `#[allow(clippy::large_enum_variant)]`
/// mirrors the same allowance on `ProbeOp`.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ProbeRequest {
    /// File-anchor verify / Seed / Rebase. The walker runs a single
    /// `lstat` and returns `ProbeOutcome::AnchorOk(LeafEntry)` (or
    /// `Vanished` on absent / kind-mismatch / `Failed { errno }` on I/O).
    /// No baseline (a leaf has no descendants to skip), no `force_walk`
    /// (the path is one syscall), no `forced` (mtime-skip is not a
    /// concept for `lstat`).
    AnchorFile {
        /// Owner the engine demuxes the response back to. Echoed back
        /// on `ProbeResponse` and used by the Sensor's expectation-map
        /// insertion.
        owner: ProbeOwner,
        /// Engine-monotonic correlation token — pairs request with response.
        correlation: ProbeCorrelation,
        /// Filesystem path of the anchor at probe-emission time.
        /// `Arc::clone` of the slot's materialised path
        /// (`tree.path_of(profile.resource)`) — a refcount bump, no
        /// rebuild.
        target_path: Arc<Path>,
    },
    /// Subtree verify / Seed / Rebase / Standard. Recursive Dir walk
    /// honouring `scan_config`. Walker returns
    /// `ProbeOutcome::SubtreeOk(Arc<DirSnapshot>)` rooted at
    /// `target_path` (or `Vanished` / `Failed`).
    Subtree {
        /// Owner the engine demuxes the response back to. Echoed back
        /// on `ProbeResponse` and used by the Sensor's expectation-map
        /// insertion.
        owner: ProbeOwner,
        /// Engine-monotonic correlation token — pairs request with response.
        correlation: ProbeCorrelation,
        /// Filesystem path of the directory to walk. `Arc::clone` of
        /// `tree.path_of(target_resource)` shipped on the wire — the
        /// walker has no `Tree` and never needs the engine's
        /// `ResourceId`. An empty path is the lone failure mode (the
        /// engine's stale-id sentinel); the walker treats empty as
        /// `Vanished`.
        target_path: Arc<Path>,
        /// `ScanConfig` to honour (recursive, hidden, exclude, pattern,
        /// `max_depth`). Cloned at emit time.
        scan_config: ScanConfig,
        /// `Profile.config_hash` at emission time. Walker stamps every
        /// `DirSnapshot.captured_with` so two Profiles sharing a Resource
        /// with different filters cannot produce identical `dir_hash` for
        /// divergent in-scope content.
        captured_with: u64,
        /// Engine's last-known view of `target_path`'s subtree. The
        /// walker consults `baseline_subtree.root_meta` for mtime-skip
        /// and propagates child baselines via each child's
        /// `DirChild::Covered(arc)`, resolved by name through
        /// [`crate::DirSnapshot::lookup_covered_dir`]. `None` means
        /// "no prior observation": first Seed of a fresh Profile.
        ///
        /// Cheap to ship — `Arc::clone` on the channel send. Multiple
        /// workers may hold the same Arc concurrently: `DirSnapshot` is
        /// fully immutable post-construction (hashes are eager fields,
        /// not a lazy cache).
        baseline_subtree: Option<Arc<DirSnapshot>>,
        /// Set of paths the walker MUST enumerate (refusing mtime-skip)
        /// at any directory whose path equals one of these OR is an
        /// ancestor of one. Populated from kqueue events that arrived
        /// since the last probe at this target.
        ///
        /// Walker checks `force_walk.iter().any(|p| p.starts_with(current))`
        /// — O(N) per directory, N = `|force_walk|` (typically 1–5). The
        /// set is *minimal* (only the dirty paths); the walker's
        /// prefix-match covers the "ancestor of forced descendant" case
        /// without engine-side closure construction.
        ///
        /// `BTreeSet` (not `Vec`) so iteration order is deterministic
        /// for replay. `Arc<Path>` entries are `Arc::clone`s of the
        /// engine's slot paths — shipped without re-allocating.
        force_walk: BTreeSet<Arc<Path>>,
        /// `true` ⇒ walker bypasses mtime-skip at every directory
        /// regardless of `baseline_subtree` and `force_walk`. Engine sets
        /// this when `PreFireBurst.forced` is true (max-settle deadline
        /// elapsed; force-fire).
        ///
        /// Defensive: mtime-skip is correct under normal semantics, but a
        /// forced probe wants the freshest possible snapshot regardless
        /// of cost.
        forced: bool,
    },
    /// Pending-descent prefix probe. Walker enumerates one level of
    /// `target_path` (no recursion, no exclude/pattern, hidden=true) and
    /// returns `ProbeOutcome::SubtreeOk(arc)` containing the prefix's
    /// direct children — descent dispatch reads `arc.entries.get(name)`
    /// and discards the snapshot (it is never spliced into
    /// `Profile.current`).
    Descent {
        /// Owner the engine demuxes the response back to. Echoed back
        /// on `ProbeResponse` and used by the Sensor's expectation-map
        /// insertion.
        owner: ProbeOwner,
        /// Engine-monotonic correlation token — pairs request with response.
        correlation: ProbeCorrelation,
        /// Filesystem path of the descent prefix at probe-emission time.
        /// The engine routes responses by `(owner, correlation)` against
        /// the owner's state-resident `ProbeSlot` (the descent prefix
        /// lives on `DescentState`; the promoter enumeration target is
        /// the `Active` slot's tag); the walker only needs the path.
        /// `Arc::clone` of the slot's materialised path — no rebuild.
        target_path: Arc<Path>,
    },
}

impl ProbeRequest {
    /// Owner the engine demuxes the response back to. Determinism-sort key for
    /// [`crate::StepOutput::probe_ops`] (via [`ProbeOp::owner`]).
    #[must_use]
    pub const fn owner(&self) -> ProbeOwner {
        match self {
            Self::AnchorFile { owner, .. }
            | Self::Subtree { owner, .. }
            | Self::Descent { owner, .. } => *owner,
        }
    }

    /// Correlation token. Used by the bin's expectation-map insertion
    /// in the sensor's `Prober::submit` (via `WorkerProber`) and by
    /// the worker's post-run cleanup. Never read by the engine after
    /// emit.
    #[must_use]
    pub const fn correlation(&self) -> ProbeCorrelation {
        match self {
            Self::AnchorFile { correlation, .. }
            | Self::Subtree { correlation, .. }
            | Self::Descent { correlation, .. } => *correlation,
        }
    }

    /// Filesystem path the walker probes. Every variant carries one;
    /// returns the borrowed path verbatim. The wire is path-keyed —
    /// this is the load-bearing identifier the walker dispatches on.
    #[must_use]
    pub fn target_path(&self) -> &Path {
        match self {
            Self::AnchorFile { target_path, .. }
            | Self::Subtree { target_path, .. }
            | Self::Descent { target_path, .. } => target_path,
        }
    }
}

/// Walker→engine probe response. Flat — `(owner, correlation)` is the
/// staleness key the engine gates against the owner's in-flight
/// `ProbeSlot`; `outcome` carries the per-variant payload.
#[derive(Debug, Clone)]
pub struct ProbeResponse {
    pub owner: ProbeOwner,
    pub correlation: ProbeCorrelation,
    pub outcome: ProbeOutcome,
}

/// Walker outcome.
///
/// Four variants, intent-agnostic on Vanished/Failed (the engine routes
/// those by `Profile.state` discriminator + pre-/post-fire phase, not by
/// request shape — a vanished anchor is a vanished anchor regardless of
/// whether the walker was looking at a file or a directory).
#[derive(Debug, Clone)]
pub enum ProbeOutcome {
    /// `AnchorFile` request returned a leaf observation. Sole producer is
    /// the walker's `probe_anchor_file`.
    AnchorOk(LeafEntry),
    /// `Subtree` *or* `Descent` request returned a directory observation.
    /// Descent and Subtree share a wire shape (`Arc<DirSnapshot>`) — what
    /// differs is the engine-side dispatch state, not the data the walker
    /// hands back. The snapshot is pure content (`root_meta`,
    /// `captured_with`, `entries`); engine-side identity stays at the
    /// dispatch layer (the owner's state-resident `ProbeSlot`).
    SubtreeOk(Arc<DirSnapshot>),
    /// Path absent (`ENOENT`) or kind mismatch (file probe found dir, dir
    /// probe found file). Routed to whichever `dispatch_*_vanished`
    /// corresponds to the Profile's state.
    Vanished,
    /// I/O error at the *root* of the probe (root `lstat`, permission,
    /// `EIO`). Mid-walk errors don't surface here — they
    /// skip-and-continue with `tracing::warn!`.
    Failed { errno: i32 },
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
    /// placeholder, post-`add_watch` before the first probe), and the
    /// sensor accepts whatever inode resolves while caching the
    /// observed kind for normalization / mask translation.
    ///
    /// `events` is the carrier for the per-Resource event-class union:
    /// the engine ships [`crate::Resource::events_union`] on every
    /// `Watch` op, the sensor diffs the cached per-FD mask, and
    /// re-registers iff different. `ClassSet::EMPTY` degrades to
    /// identity-floor-only delivery (kqueue: `NOTE_DELETE | NOTE_RENAME
    /// | NOTE_REVOKE`).
    Watch {
        resource: ResourceId,
        path: Arc<Path>,
        kind: ResourceKind,
        events: ClassSet,
    },
    Unwatch {
        resource: ResourceId,
    },
}

impl WatchOp {
    /// The Resource this op targets. Every variant carries one — the
    /// match is exhaustive and `const`. This is the determinism-sort
    /// key for [`crate::StepOutput::watch_ops`].
    #[must_use]
    pub const fn resource(&self) -> ResourceId {
        match self {
            Self::Watch { resource, .. } | Self::Unwatch { resource } => *resource,
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

// `ProbeRequest::Subtree` carries baseline_subtree / force_walk / forced
// etc., so `Probe` dwarfs `Cancel`. Boxing it would add an allocation
// per probe (every burst emits one `Probe`; `Cancel` is sparse) for no
// gain: a `ProbeOp` lives in a `StepOutput::probe_ops` BTreeMap node,
// heap-allocated on insert regardless of variant width — the size never
// rides an inline slot. Accept the delta explicitly.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ProbeOp {
    Probe { request: ProbeRequest },
    Cancel { owner: ProbeOwner },
}

impl ProbeOp {
    /// The owner this op addresses. Both variants carry one (the
    /// `Probe` variant via its nested [`ProbeRequest`]). This is the
    /// determinism-sort key for [`crate::StepOutput::probe_ops`].
    #[must_use]
    pub const fn owner(&self) -> ProbeOwner {
        match self {
            Self::Probe { request } => request.owner(),
            Self::Cancel { owner } => *owner,
        }
    }
}
