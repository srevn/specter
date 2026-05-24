//! `InotifyWatcher` — inotify-backed `FsWatcher` impl.
//!
//! Single-threaded: one thread owns the [`InotifyWatcher`] value and
//! drives [`FsWatcher::watch`] / [`FsWatcher::unwatch`] between
//! [`FsWatcher::poll_until`] calls. The wake handle
//! ([`InotifyWakeHandle`]) is the only cross-thread surface — see
//! [`crate::inotify::wake`] for the lifecycle discipline.
//!
//! Mirror of [`crate::kqueue::watcher::KqueueWatcher`] with the inotify
//! substrate (eventfd in place of `EVFILT_USER`, epoll multiplex over
//! `(inotify_fd, wake_fd)` in place of kqueue's single-fd kernel filter
//! set).
//!
//! # Drop semantics
//!
//! Default field-order drop:
//! - `inotify_fd` drops first → the kernel reaps every per-watch
//!   descriptor on this instance and queues the corresponding
//!   `IN_IGNORED` records (which no consumer reads; benign).
//! - `wake_fd` (`Arc`) decrements; if the last clone, the eventfd
//!   closes. Wake handles holding clones outlive the watcher and a
//!   `wake()` from those becomes a no-op-equivalent (no consumer
//!   drains the resulting counter), with no UB.
//! - `epoll_fd` drops last → the epoll instance closes; the kernel had
//!   already removed the inotify_fd / wake_fd registrations as those
//!   fds closed.
//!
//! # Per-resource entry cache
//!
//! Each entry caches `(wd, mask, kind)`: the watch descriptor returned
//! by `inotify_add_watch`, the kernel-side mask we last installed, and
//! the fstat-verified inode shape (established at fresh-watch time;
//! refreshed only on the slow path of [`InotifyWatcher::rewatch_inner`]
//! — see its docstring). The triple is stored as a single struct (not
//! three parallel maps keyed by `ResourceId`) so every install path
//! writes it atomically and every teardown clears it atomically — the
//! "forgot to update one of three maps" failure mode is
//! unrepresentable. Mirror of kqueue's `KqueueEntry { fd, fflags, kind }`.
//!
//! A re-`watch()` with an unchanged mask short-circuits without a
//! kernel install. The fast path (cached kind is `File` / `Dir`)
//! short-circuits with **zero syscalls** — the cache predicts the
//! install mask from `events × cached_kind` and the kernel's "replace
//! mask" semantics on an existing path would produce identical bits.
//! The slow path (cached kind is `Unknown`, the rare socket / fifo /
//! device case) reopens and `fstat`s before the short-circuit check
//! because the Unknown-collapse mask cannot be trusted to match the
//! live inode shape. See [`InotifyWatcher::rewatch_inner`].

use crate::inotify::wake::InotifyWakeHandle;
use crate::inotify::{ffi, normalize, record, translate};
use crate::{FsWatcher, WakeHandle, WatchFailure, WatchFailureExt, WatcherEvent};
use slotmap::SecondaryMap;
use specter_core::{ClassSet, FsEvent, OverflowScope, ResourceId, ResourceKind};
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::os::fd::OwnedFd;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Token tagging the inotify fd in epoll. The `poll_until` consumer
/// reads `epoll_event.u64` to discriminate inotify-data-ready from
/// wake-fired; distinct from [`WAKE_TOKEN`].
const INOTIFY_TOKEN: u64 = 0xDEAD_BEEF_DEAD_BEEF;

/// Token tagging the wake (eventfd) in epoll. Distinct from
/// [`INOTIFY_TOKEN`]; recognisable in debug output.
const WAKE_TOKEN: u64 = 0xCAFE_BABE_CAFE_BABE;

/// Drain buffer size in bytes. Per `inotify(7)`, the per-event minimum
/// is `sizeof(struct inotify_event) + NAME_MAX + 1` ≈ 273 bytes; 16 KiB
/// drains a typical event burst in one `read()` syscall and is well
/// above the floor (the kernel returns `EINVAL` on a buffer too small
/// for the next record).
const READ_BUF_BYTES: usize = 16 * 1024;

#[derive(Debug)]
pub struct InotifyWatcher {
    /// Single inotify fd for all watches. Owned exclusively by the
    /// watcher; close ⇒ kernel auto-removes every per-watch descriptor
    /// (per `inotify(7)`). Plain [`OwnedFd`] (no `Arc`) — only the
    /// watcher's owning thread reads from it; the wake handle uses the
    /// separate `wake_fd` eventfd.
    inotify_fd: OwnedFd,

    /// Eventfd for cross-thread wake. `Arc` so wake handles can hold
    /// their own clones without borrowing from the watcher; drop of the
    /// last clone closes the fd. See [`InotifyWakeHandle`] for the
    /// lifecycle discipline.
    wake_fd: Arc<OwnedFd>,

    /// Epoll fd watching `(inotify_fd, wake_fd)`. Owned, not Arc'd —
    /// only `poll_until` reads from it; wake handles never touch it.
    epoll_fd: OwnedFd,

    /// `ResourceId → (wd, mask, kind)`. Populated by `watch()` on
    /// successful install, cleared by `unwatch()`. The mask cache lets
    /// a re-`watch()` skip the syscall when the install mask is
    /// unchanged (mirror of kqueue's `fflags` discipline); the kind
    /// cache is consumed by
    /// [`crate::inotify::normalize::mask_to_fs_event`] for File-vs-Dir
    /// disambiguation. See [`InotifyEntry`] for the field-level
    /// lifecycle.
    by_resource: SecondaryMap<ResourceId, InotifyEntry>,

    /// `wd → ResourceId`. inotify events don't carry userdata
    /// (kqueue's `udata` analogue), so the watcher pays the storage
    /// to route a record's `wd` back to the slot it belongs to. wd
    /// values are dense small integers; at typical watch counts
    /// `BTreeMap`'s `O(log n)` lookup sits well below the per-event
    /// syscall cost (`read` / `inotify_add_watch` dominate by orders
    /// of magnitude), and deterministic iteration is free for the
    /// drain-time tracing.
    by_wd: BTreeMap<libc::c_int, ResourceId>,

    /// `wd`s in the "draining" state: `inotify_rm_watch` has been
    /// called but the kernel's `IN_IGNORED` for that wd has not yet
    /// arrived in our read buffer. Events on draining wds are dropped
    /// during `poll_until`; the `IN_IGNORED` consumption reaps the
    /// flag. This closes a wd-reuse race — a subsequent
    /// `inotify_add_watch` may return the same wd before userspace
    /// observes the `IN_IGNORED`, and pre-rm events on the old inode
    /// would otherwise mis-attribute to the freshly attached resource.
    draining_wds: BTreeSet<libc::c_int>,

    /// Drain buffer for inotify event records. Sized at construction
    /// and reused across drains — `poll_once` performs no allocation
    /// on the hot path.
    read_buf: Vec<u8>,
    /// Per-drain dedup horizon. Cleared at the start of every
    /// [`Self::poll_once`] call so the kernel's `IN_MODIFY` +
    /// `IN_CLOSE_WRITE` pair (both normalising to
    /// [`FsEvent::Modified`] for the same resource) collapses to one
    /// emitted event.
    ///
    /// `Vec` rather than `BTreeSet`: a typical burst yields ~20–50
    /// distinct `(ResourceId, FsEvent)` pairs, well inside the
    /// "linear scan beats logarithmic node-walk" regime — the
    /// comparisons are cache-resident tuple compares. The
    /// load-bearing property is [`Vec::clear`]'s capacity-preserving
    /// reset: across the watcher's lifetime, only the first few
    /// drains pay any allocation cost; subsequent calls reuse the
    /// stabilised buffer.
    seen: Vec<(ResourceId, FsEvent)>,
}

/// Per-resource cached install state — the `(wd, mask, kind)` triple
/// installed at fresh-watch time and updated atomically on rewatch.
///
/// - `wd` is the watch descriptor returned by `inotify_add_watch`.
///   Stable across same-inode re-installs (the kernel's "replace mask"
///   semantics return the same wd); changes on inode-swap — see the
///   `wd != prior.wd` branch in the re-watch path.
/// - `mask` is the exact bits last passed to `inotify_add_watch`,
///   including install-time directional flags like `IN_ONLYDIR` for
///   Dir watches. A re-`watch()` recomputes the mask from the user's
///   `events` set and the cached / observed kind; an unchanged mask
///   short-circuits without a kernel install (zero syscalls on the
///   fast path, one open + fstat pair on the slow path — see
///   [`InotifyWatcher::rewatch_inner`]).
/// - `kind` is the fstat-verified inode shape, closing the TOCTOU
///   window between the engine's `WatchOp::Watch.kind` and the kernel's
///   path-resolution at install time. Established at fresh-watch time
///   and refreshed only on the slow path of
///   [`InotifyWatcher::rewatch_inner`] — the rare case where the
///   fresh-watch fstat returned `Unknown` (socket / fifo / device
///   target). The fast path treats the cached kind as invariant: an
///   inode swap surfaces as `IN_DELETE_SELF` / `IN_UNMOUNT` on the
///   prior wd, the watcher's `poll_until` clears `by_resource[r]` via
///   the `IN_IGNORED` cleanup arm, and the engine's next `Watch`
///   re-runs the open + fstat + classify chain through fresh-watch.
///   Consumed by [`crate::inotify::normalize::mask_to_fs_event`] to
///   disambiguate `IN_MODIFY` on Dir vs File defensive paths.
///
/// Stored as a single struct so the triple is atomic: every install
/// path writes all three fields together, every teardown clears them
/// together. Mirror of kqueue's `KqueueEntry { fd, fflags, kind }` —
/// inotify's `kind` is *almost* invariant the way kqueue's strictly is;
/// the asymmetry tracks the underlying substrate (inotify reopens per
/// rewatch via `/proc/self/fd/N`; kqueue binds an fd per resource).
#[derive(Debug, Clone, Copy)]
struct InotifyEntry {
    wd: libc::c_int,
    mask: u32,
    kind: ResourceKind,
}

impl InotifyWatcher {
    /// Create a fresh inotify instance, eventfd, and epoll instance,
    /// and register the inotify and wake fds on the epoll under
    /// distinct tokens.
    ///
    /// Returns the syscall error on any step's failure — `EMFILE` /
    /// `ENFILE` / `ENOMEM` are the only realistic cases on the init
    /// trio (`inotify_init1` / `eventfd` / `epoll_create1`); `EBADF`
    /// from `epoll_ctl` is structurally unreachable because both
    /// argument fds were just created by the helpers above. The bin
    /// treats startup failures as fatal — symmetric with the kqueue
    /// branch's behaviour when its own `kqueue_new` fails.
    ///
    /// Drop order on a partial failure: each `?` propagates the error,
    /// and any [`OwnedFd`] already bound to a local drops via RAII so
    /// the kernel reaps every fd this constructor opened. No leak is
    /// possible.
    pub fn new() -> io::Result<Self> {
        let inotify_fd = ffi::inotify_init()?;
        let wake_fd = Arc::new(ffi::eventfd_create()?);
        let epoll_fd = ffi::epoll_create()?;

        ffi::epoll_register(&epoll_fd, &inotify_fd, INOTIFY_TOKEN)?;
        ffi::epoll_register(&epoll_fd, &wake_fd, WAKE_TOKEN)?;

        Ok(Self {
            inotify_fd,
            wake_fd,
            epoll_fd,
            by_resource: SecondaryMap::new(),
            by_wd: BTreeMap::new(),
            draining_wds: BTreeSet::new(),
            read_buf: vec![0u8; READ_BUF_BYTES],
            seen: Vec::new(),
        })
    }

    /// Internal `watch` body — dispatches by entry presence. The trait
    /// wrapper maps the inner `io::Error` set into a typed
    /// [`WatchFailure`] at the boundary so `?` propagation across the
    /// open / fstat / add_watch chain stays uniform.
    ///
    /// - `by_resource[r]` populated → [`Self::rewatch_inner`]
    ///   (re-register on the same `ResourceId`, with inode-swap
    ///   detection).
    /// - `by_resource[r]` empty → [`Self::fresh_watch_inner`] (open +
    ///   fstat verify + install).
    ///
    /// The engine-supplied `kind` is structurally irrelevant on
    /// rewatch (both backends ignore it; the cached / observed value
    /// is the source of truth). The split makes this explicit at the
    /// call boundary — [`Self::rewatch_inner`] does not take `kind`.
    fn watch_inner(
        &mut self,
        r: ResourceId,
        path: &Path,
        kind: ResourceKind,
        events: ClassSet,
    ) -> io::Result<()> {
        if self.by_resource.contains_key(r) {
            self.rewatch_inner(r, path, events)
        } else {
            self.fresh_watch_inner(r, path, kind, events)
        }
    }

    /// Re-register `r`'s entry against the path. Engine triggers this
    /// when `Resource.events_union` changes at non-zero refcount.
    ///
    /// Splits on the cached kind:
    ///
    /// - **Fast path** (`cached_kind ∈ { File, Dir }`). The cache is
    ///   structurally reliable: an inode swap under the path queues
    ///   `IN_DELETE_SELF` / `IN_UNMOUNT` on the prior wd, the next
    ///   `poll_until` drain reaps `by_resource[r]` via the
    ///   `IN_IGNORED` cleanup arm, and a subsequent `Watch` re-enters
    ///   through `fresh_watch_inner` — so re-entry to *this* function
    ///   means the cached kind still matches the live inode. The
    ///   install mask is predictable from `events × cached_kind`; an
    ///   unchanged mask short-circuits with **zero syscalls**. A
    ///   changed mask reopens to detect the rare same-kind atomic
    ///   rename (the wd-swap window between unlink and the drain),
    ///   verifies the kind via `matches_or_unknown` (a kind flip
    ///   surfaces as `ENOTDIR` so the engine reseeds via descent),
    ///   and commits.
    ///
    /// - **Slow path** (`cached_kind == Unknown`). Fresh-watch's fstat
    ///   classified the inode as non-Dir / non-regular (socket / fifo
    ///   / device). The Unknown-collapse install mask is File-shape
    ///   (via [`ResourceKind::effective`]); if an inode swap mutates
    ///   the path to a Dir / File without an intervening drain,
    ///   predicting the mask from `cached_kind == Unknown` would
    ///   silently install File-shape bits on a Dir (missing
    ///   `IN_CREATE` / `IN_DELETE` / `IN_MOVED_*` / `IN_ONLYDIR`).
    ///   This branch always reopens + fstats first, derives the
    ///   install mask from the observed kind, and refreshes the
    ///   cached kind on commit. No `matches_or_unknown` gate —
    ///   `Unknown` is the wildcard.
    ///
    /// Doesn't take the engine-supplied `kind` — the cached / observed
    /// value is the source of truth on rewatch (kqueue's rewatch is
    /// the same shape by construction; inotify mirrors it).
    ///
    /// # Precondition
    ///
    /// `by_resource[r]` must be populated. The dispatcher
    /// [`Self::watch_inner`] enforces this; calling this directly with
    /// an empty entry panics.
    fn rewatch_inner(&mut self, r: ResourceId, path: &Path, events: ClassSet) -> io::Result<()> {
        let prior = self
            .by_resource
            .get(r)
            .copied()
            .expect("rewatch_inner invoked without existing entry");
        let cached_kind = prior.kind;

        if matches!(cached_kind, ResourceKind::Unknown) {
            return self.rewatch_slow_unknown(r, path, events, prior);
        }
        self.rewatch_fast_known(r, path, events, prior)
    }

    /// Fast-path rewatch for `cached_kind ∈ { File, Dir }`. Predicts
    /// the install mask from the cache; short-circuits with zero
    /// syscalls on no diff. See [`Self::rewatch_inner`] for the
    /// branch's structural invariants.
    fn rewatch_fast_known(
        &mut self,
        r: ResourceId,
        path: &Path,
        events: ClassSet,
        prior: InotifyEntry,
    ) -> io::Result<()> {
        let cached_kind = prior.kind;
        let install_mask = compute_install_mask(events, cached_kind);
        if install_mask == prior.mask {
            tracing::trace!(
                ?r,
                ?events,
                mask = format_args!("{install_mask:#x}"),
                "inotify re-watch noop (mask unchanged)"
            );
            return Ok(());
        }

        // Reopen + fstat. Verifies the inode shape hasn't mutated
        // under the path during the wd-swap window (rare; bounded by
        // the bin's drain cadence). `matches_or_unknown` enforces
        // `observed == cached_kind` here — the `Unknown` wildcard
        // self-branch is excluded by this function's entry condition,
        // so the gate trips iff the inode actually flipped kind. A
        // disagreement maps to `ENOTDIR` (path-fatal); the engine
        // reseeds via descent rather than installing a kind-incoherent
        // watch.
        let fd = ffi::open_o_path(path)?;
        let observed_kind = ffi::fstat_kind(&fd)?;
        if !cached_kind.matches_or_unknown(observed_kind) {
            tracing::warn!(
                ?r,
                ?path,
                expected = ?cached_kind,
                observed = ?observed_kind,
                "inotify re-watch kind mismatch — cached != fstat"
            );
            return Err(io::Error::from_raw_os_error(libc::ENOTDIR));
        }

        // Gate passed ⇒ `observed_kind == cached_kind` (gate excludes
        // the Unknown-wildcard arm on this branch), so the predicted
        // `install_mask` is the install mask. Cache stays at
        // `cached_kind`.
        self.commit_rewatch(r, &fd, install_mask, cached_kind, prior)
    }

    /// Slow-path rewatch for `cached_kind == Unknown`. Always reopens
    /// and fstats to derive the install mask from the live inode
    /// shape; refreshes the cached kind on commit. See
    /// [`Self::rewatch_inner`] for why the fast-path optimization is
    /// unsound here.
    fn rewatch_slow_unknown(
        &mut self,
        r: ResourceId,
        path: &Path,
        events: ClassSet,
        prior: InotifyEntry,
    ) -> io::Result<()> {
        let fd = ffi::open_o_path(path)?;
        let observed_kind = ffi::fstat_kind(&fd)?;
        let install_mask = compute_install_mask(events, observed_kind);

        // Genuine noop: mask unchanged AND the inode is still
        // non-classifiable. Skip the kernel install; the cache stays
        // at Unknown so the next rewatch re-checks. The mask-equal
        // check alone is not sufficient — a flip from Unknown to File
        // can produce the same File-shape mask but must still refresh
        // the cached kind so future rewatches take the fast path.
        if install_mask == prior.mask && matches!(observed_kind, ResourceKind::Unknown) {
            tracing::trace!(
                ?r,
                ?events,
                mask = format_args!("{install_mask:#x}"),
                "inotify re-watch noop (mask unchanged; cached kind still Unknown)"
            );
            return Ok(());
        }
        self.commit_rewatch(r, &fd, install_mask, observed_kind, prior)
    }

    /// Shared rewatch commit: install via
    /// [`ffi::inotify_add_watch_fd`] (the fused `O_PATH` →
    /// `/proc/self/fd/N` → `inotify_add_watch` helper), detect wd-swap
    /// (atomic rename of the path between prior install and this
    /// re-add), guard against hardlink aliasing, then atomically
    /// replace `r`'s [`InotifyEntry`].
    ///
    /// `kind` is the value to store in the refreshed entry — the
    /// fast path passes the (invariant) `prior.kind`; the slow path
    /// passes the freshly observed kind.
    ///
    /// Failure shape: `inotify_add_watch_fd` errors propagate before
    /// any state mutation. A successful add followed by
    /// hardlink-aliasing rejection routes through
    /// [`Self::reject_aliased_install`], which restores the aliased
    /// resource's mask and clears `r`'s entry — the engine sees a
    /// clean unwatched state for `r`.
    fn commit_rewatch(
        &mut self,
        r: ResourceId,
        fd: &OwnedFd,
        install_mask: u32,
        kind: ResourceKind,
        prior: InotifyEntry,
    ) -> io::Result<()> {
        let wd = ffi::inotify_add_watch_fd(&self.inotify_fd, fd, install_mask)?;

        // Inode-swap detection. A different wd means the path now
        // resolves to a different inode (atomic rename swapped the
        // path between the prior install and this re-add). Mark the
        // prior wd as draining so any pre-rm events on it are dropped
        // from the next `poll_until` iteration — the kernel's
        // `IN_IGNORED` arrives later in the drain stream and reaps
        // the flag.
        if wd != prior.wd {
            tracing::debug!(
                ?r,
                old_wd = prior.wd,
                new_wd = wd,
                "inotify rewatch resolved to different inode"
            );
            self.draining_wds.insert(prior.wd);
            self.by_wd.remove(&prior.wd);
            if let Err(e) = ffi::inotify_rm_watch(&self.inotify_fd, prior.wd) {
                // EINVAL ⇒ kernel already reaped the wd (the old
                // inode was deleted out from under us). The
                // `IN_IGNORED` was queued synchronously and will
                // arrive on the drain stream; `draining_wds` covers
                // the gap.
                if e.raw_os_error() != Some(libc::EINVAL) {
                    tracing::warn!(
                        ?r,
                        wd = prior.wd,
                        error = ?e,
                        "inotify_rm_watch on prior (inode-swap) failed"
                    );
                }
            }
        }

        // Hardlink aliasing guard — see [`Self::reject_aliased_install`].
        if let Some(&existing) = self.by_wd.get(&wd)
            && existing != r
        {
            self.reject_aliased_install(existing, r, wd, fd);
            return Err(io::Error::from_raw_os_error(libc::EEXIST));
        }

        // Commit the new (wd, mask, kind) triple atomically.
        self.by_resource.insert(
            r,
            InotifyEntry {
                wd,
                mask: install_mask,
                kind,
            },
        );
        self.by_wd.insert(wd, r);
        tracing::debug!(
            ?r,
            wd,
            old_mask = format_args!("{:#x}", prior.mask),
            new_mask = format_args!("{install_mask:#x}"),
            old_kind = ?prior.kind,
            new_kind = ?kind,
            "inotify rewatch (mask or kind changed)"
        );
        Ok(())
    }

    /// First-install for `r` on the 0→1 `watch_demand` edge.
    ///
    /// Race-free install via [`ffi::open_o_path`] + `/proc/self/fd/N`:
    /// the fd binds to a specific inode, and `inotify_add_watch` on
    /// the magic-symlink path resolves to that inode regardless of
    /// intervening renames at `path`. The fstat verification then
    /// matches the engine's expected `kind`; a kind disagreement maps
    /// to `ENOTDIR`, which the trait wrapper classifies as
    /// [`WatchFailure::Resource`] so the engine routes through the
    /// path-fatal recovery channel.
    ///
    /// On hardlink aliasing (a second `ResourceId` resolving to the
    /// same inode as an existing entry), the kernel returns the
    /// already-allocated wd; the install is rejected and
    /// [`Self::reject_aliased_install`] restores the existing
    /// resource's mask.
    ///
    /// # Precondition
    ///
    /// `by_resource[r]` must be empty. The dispatcher
    /// [`Self::watch_inner`] enforces this.
    fn fresh_watch_inner(
        &mut self,
        r: ResourceId,
        path: &Path,
        kind: ResourceKind,
        events: ClassSet,
    ) -> io::Result<()> {
        // 1) Open with `O_PATH | O_NOFOLLOW`. The fd binds to a
        //    specific inode regardless of subsequent renames at
        //    `path`. `O_PATH` permits `fstat` even without read
        //    permission and does not pin the inode against `unlink`
        //    — exactly the discipline kqueue's `O_EVTONLY` provides
        //    on Darwin.
        let fd = ffi::open_o_path(path)?;

        // 2) `fstat` the fd. Race-stable kind discovery — the fd
        //    binds to a single inode whose kind cannot mutate
        //    underneath us.
        let observed_kind = ffi::fstat_kind(&fd)?;

        // 3) Verify against the engine's expectation. Unknown is a
        //    wildcard; otherwise the kinds must agree.
        if !kind.matches_or_unknown(observed_kind) {
            tracing::warn!(
                ?r,
                ?path,
                expected = ?kind,
                observed = ?observed_kind,
                "inotify watch kind mismatch — engine expected != fstat"
            );
            return Err(io::Error::from_raw_os_error(libc::ENOTDIR));
        }

        // 4) Compute the install mask using the verified kind.
        //    `compute_install_mask` ORs `IN_ONLYDIR` for Dir watches
        //    as defense-in-depth — the fstat already confirmed
        //    Dir-ness and the `/proc/self/fd/N` install is race-free,
        //    but the kernel-side guard is a free belt-and-braces
        //    safety net.
        let install_mask = compute_install_mask(events, observed_kind);

        // 5) Install via the fused FFI helper. The kernel's procfs
        //    resolver returns the exact inode `fd` refers to, closing
        //    the TOCTOU window between fstat and add_watch that a
        //    naive `inotify_add_watch(path)` would leave open.
        let wd = ffi::inotify_add_watch_fd(&self.inotify_fd, &fd, install_mask)?;

        // 6) Hardlink aliasing guard — see
        //    [`Self::reject_aliased_install`]. `fd` is intentionally
        //    still alive here so the helper can re-use the same
        //    `O_PATH` fd for the kernel-side mask restoration on the
        //    existing resource's behalf.
        if let Some(&existing) = self.by_wd.get(&wd)
            && existing != r
        {
            self.reject_aliased_install(existing, r, wd, &fd);
            return Err(io::Error::from_raw_os_error(libc::EEXIST));
        }

        // 7) Commit the (wd, mask, kind) triple atomically. `fd`
        //    drops at end of scope; the kernel watches the inode the
        //    fd resolved to at `add_watch` time, independent of fd
        //    lifetime (per `inotify(7)`).
        self.by_resource.insert(
            r,
            InotifyEntry {
                wd,
                mask: install_mask,
                kind: observed_kind,
            },
        );
        self.by_wd.insert(wd, r);
        tracing::debug!(
            ?r,
            ?path,
            kind = ?observed_kind,
            ?events,
            wd,
            mask = format_args!("{install_mask:#x}"),
            "inotify watch"
        );
        Ok(())
    }

    /// Common epilogue for the two hardlink-aliasing rejection sites
    /// (one in [`Self::fresh_watch_inner`], one in
    /// [`Self::rewatch_inner`]).
    ///
    /// Two `ResourceId`s pointing to the same inode receive the same
    /// `wd` from the kernel — there is one kernel-side watch entry
    /// per `(inotify_fd, inode)` pair (per `inotify(7)`'s "the
    /// existing watch is updated" semantics). v1 rejects the second
    /// attachment, but the just-issued `inotify_add_watch` has
    /// already clobbered the existing watch's mask with the new
    /// resource's requested mask (the kernel's "replace mask"
    /// semantics). A naive `inotify_rm_watch` would tear down the
    /// existing resource's kernel-side registration entirely; the
    /// safe recovery is to restore the existing resource's mask via
    /// a follow-up [`ffi::inotify_add_watch_fd`] on the same
    /// `O_PATH` fd. The restoration is best-effort; on failure the
    /// existing resource's mask remains the rejected resource's
    /// requested mask until its next reconcile triggers a re-add —
    /// a documented v1 limitation, not a correctness regression.
    ///
    /// After the mask-restore, this helper clears `r`'s entry so the
    /// engine sees a clean "unwatched-and-unknown" state on retry.
    /// The mask-restore and removal steps are separate so the borrow
    /// checker tolerates the intermediate `&self.by_resource`
    /// immutable borrow before the `self.by_resource.remove(r)`
    /// mutable borrow.
    ///
    /// `fd` is the caller's `O_PATH` fd for the alias-causing path;
    /// the caller must keep it alive across this call (the kernel
    /// resolves the magic-symlink path at syscall time inside
    /// [`ffi::inotify_add_watch_fd`], and a closed fd would surface
    /// as `ENOENT`).
    fn reject_aliased_install(
        &mut self,
        existing: ResourceId,
        r: ResourceId,
        wd: libc::c_int,
        fd: &OwnedFd,
    ) {
        match self.by_resource.get(existing).copied() {
            Some(existing_entry) => {
                if let Err(e) = ffi::inotify_add_watch_fd(&self.inotify_fd, fd, existing_entry.mask)
                {
                    tracing::warn!(
                        ?r,
                        ?existing,
                        wd,
                        error = ?e,
                        "inotify mask restore failed after hardlink aliasing rejection \
                         — existing resource's kernel-side mask is now this resource's \
                         requested mask (v1 limitation)"
                    );
                }
            }
            None => {
                tracing::error!(
                    ?r,
                    ?existing,
                    wd,
                    "by_wd[wd] = existing but by_resource[existing] missing — \
                     watcher state inconsistent; mask restoration skipped"
                );
            }
        }

        // r either had no install (fresh-watch) or had its prior wd
        // drained (re-watch with inode swap). No new watch is bound
        // to r — remove the entry so the engine's clamp → re-resolve
        // flow starts from a clean slate. `remove` is idempotent on
        // the fresh-watch case where no entry was ever inserted.
        self.by_resource.remove(r);
    }
}

impl FsWatcher for InotifyWatcher {
    /// Trait wrapper around [`Self::watch_inner`]: classifies the inner
    /// `io::Error` into a typed [`WatchFailure`] at the boundary so the
    /// engine demuxes on the variant rather than on raw errno values.
    /// The errno-name match lives in
    /// [`crate::WatchFailureExt::from_io`]; the inotify-specific
    /// vocabulary stops at this seam.
    fn watch(
        &mut self,
        r: ResourceId,
        path: &Path,
        kind: ResourceKind,
        events: ClassSet,
    ) -> Result<(), WatchFailure> {
        self.watch_inner(r, path, kind, events)
            .map_err(|e| WatchFailure::from_io(&e))
    }

    /// Tear down `r`'s kernel-side registration with the wd-reuse race
    /// mitigation: the wd is marked **draining** BEFORE the
    /// `inotify_rm_watch` so any pre-existing events on it are dropped
    /// from the next `poll_until` iteration; the kernel's synchronous
    /// `IN_IGNORED` arrives later in the drain stream and reaps the
    /// flag.
    ///
    /// Idempotent on stale ids — the early-return guard short-circuits
    /// when `by_resource` has no entry. The `(wd, mask, kind)` triple
    /// lives in one struct, so a single `remove(r)` covers all three
    /// pieces; the kernel-side `rm_watch` only runs when there was an
    /// entry to remove.
    ///
    /// `EINVAL` from `rm_watch` is benign: the kernel had already
    /// reaped the wd (the inode was deleted out from under us) and
    /// queued `IN_IGNORED` synchronously at that time. The
    /// `draining_wds` flag still covers the window — any pre-deletion
    /// events queued before the kernel's reap are dropped, and the
    /// `IN_IGNORED` consumption clears the flag.
    fn unwatch(&mut self, r: ResourceId) {
        let Some(entry) = self.by_resource.remove(r) else {
            // Stale id — no entry, no kernel-side work to do.
            return;
        };

        let wd = entry.wd;

        // Mark the wd as draining BEFORE `rm_watch`. The kernel queues
        // `IN_IGNORED` synchronously at rm_watch time; pre-existing
        // events on `wd` that haven't reached our drain buffer yet are
        // stale (the inode is no longer the ResourceId's intent), and
        // a subsequent `inotify_add_watch` from a fresh `watch()` call
        // may return the same `wd` for an unrelated inode before our
        // drain consumes the `IN_IGNORED`. The flag drops every event
        // on `wd` until the `IN_IGNORED` arrives and reaps it.
        self.draining_wds.insert(wd);

        // Drop the `by_wd` entry NOW. A subsequent `watch()` on the
        // same kernel-reused wd installs a fresh `by_wd[wd]` mapping
        // with the new ResourceId; the draining flag still holds, so
        // stale events on the old inode drop until `IN_IGNORED`
        // clears it.
        self.by_wd.remove(&wd);

        if let Err(e) = ffi::inotify_rm_watch(&self.inotify_fd, wd) {
            // EINVAL ⇒ kernel had already reaped the wd (the inode
            // was deleted before our explicit `rm_watch`). The
            // `IN_IGNORED` was queued at deletion time and will
            // arrive on the drain stream; `draining_wds` covers the
            // gap. Anything else is unexpected and worth a warn —
            // EBADF on inotify_fd, for instance, would be a
            // structural break.
            if e.raw_os_error() != Some(libc::EINVAL) {
                tracing::warn!(
                    ?r,
                    wd,
                    error = ?e,
                    "inotify_rm_watch failed (non-EINVAL); kernel-side state may leak"
                );
            }
        }

        tracing::debug!(?r, wd, "inotify unwatch");
    }

    /// One blocking drain to the engine's deadline; see
    /// [`Self::poll_once`] for the per-call mechanics (per-record demux,
    /// wake handling, `seen` dedup horizon). The trait wrapper holds
    /// no additional state — the watcher does no event coalescing of
    /// its own; the engine's settle-timer reschedule debounces above
    /// the trait boundary.
    fn poll_until(
        &mut self,
        deadline: Option<Instant>,
        out: &mut Vec<WatcherEvent>,
    ) -> Result<usize, WatchFailure> {
        self.poll_once(deadline, out)
    }

    /// Capture a wake handle. Clones the watcher's `Arc<OwnedFd>` of
    /// the eventfd so the handle survives the watcher's drop without
    /// UB (see [`InotifyWakeHandle`] for the `Arc` discipline). Cheap
    /// (one `Arc` increment + one `Box` allocation); idempotent —
    /// multiple handles coexist freely.
    fn wake_handle(&self) -> Box<dyn WakeHandle> {
        Box::new(InotifyWakeHandle::new(Arc::clone(&self.wake_fd)))
    }
}

impl InotifyWatcher {
    /// One full drain pass: clear the per-drain dedup horizon,
    /// `epoll_wait` (bounded by `deadline`), optionally drain the
    /// wake-fd counter, optionally `read_inotify` with per-record
    /// demux into `out`. Returns the count of [`WatcherEvent`]s pushed
    /// this call.
    ///
    /// **Dedup horizon ([`Self::seen`]).** Cleared at entry, then
    /// scanned per emitted record so the kernel's `IN_MODIFY` +
    /// `IN_CLOSE_WRITE` pair (both normalising to
    /// [`FsEvent::Modified`] for the same resource) collapses to one
    /// emitted event. First-write wins; the natural FIFO order from
    /// the kernel matches the emission order.
    ///
    /// **Wake handling.** `epoll_wait` distinguishes wake-fired from
    /// inotify-data-ready via the per-fd token. A wake-only return (no
    /// inotify data) drains the eventfd counter, returns `Ok(0)` so
    /// the bin's drain loop re-checks pending `WatchOp`s + shutdown
    /// flag. Concurrent wakes accumulate in the eventfd counter; one
    /// drain consumes them all atomically.
    ///
    /// **Per-record demux.** Each record routes by mask:
    ///
    /// 1. **`IN_Q_OVERFLOW`** — kernel signal that the per-instance
    ///    event queue overflowed. Lift to [`WatcherEvent::Overflow`]
    ///    with [`OverflowScope::Global`]; the engine reseeds every
    ///    in-scope Profile.
    /// 2. **`IN_IGNORED`** — kernel cleanup signal; the wd is being
    ///    reaped. Two legitimate paths reach here, distinguished by
    ///    `draining_wds` membership: a watcher-initiated
    ///    `inotify_rm_watch` (the flag is cleared here), or a
    ///    kernel-side spontaneous reap (the watched inode was
    ///    deleted/unmounted — clear per-resource state so the
    ///    engine's eventual `Unwatch` finds a clean slate).
    /// 3. **Stale event on a draining wd** — pre-rm events on a wd
    ///    whose `IN_IGNORED` hasn't arrived yet. Dropped to close the
    ///    wd-reuse race with a subsequent `inotify_add_watch`.
    /// 4. **Normal event** — resolve `wd → ResourceId`, normalize via
    ///    [`normalize::mask_to_fs_event`], dedupe via `self.seen`,
    ///    push as [`WatcherEvent::Fs`].
    ///
    /// **Errors.** Syscall failures classify through
    /// [`WatchFailureExt::from_io`] at the trait boundary. `EINTR` is
    /// retried inside the FFI helpers; the bin treats a non-`EINTR`
    /// failure as terminal for the watcher thread.
    fn poll_once(
        &mut self,
        deadline: Option<Instant>,
        out: &mut Vec<WatcherEvent>,
    ) -> Result<usize, WatchFailure> {
        // Reset the dedup horizon. `Vec::clear` preserves capacity, so
        // only the first few drains pay any allocation cost; subsequent
        // calls reuse the stabilised buffer. See [`Self::seen`] for the
        // sizing rationale.
        self.seen.clear();

        // Two slots — one per epoll-registered fd. Both can be ready
        // at once (deadline + a concurrent wake + inotify data).
        // Deadline tracking (including `EINTR`-retry remaining-budget
        // recompute) lives inside `epoll_wait`.
        let mut epoll_events = [libc::epoll_event { events: 0, u64: 0 }; 2];
        let n_ready = ffi::epoll_wait(&self.epoll_fd, &mut epoll_events, deadline)
            .map_err(|e| WatchFailure::from_io(&e))?;

        if n_ready == 0 {
            // Timeout — caller's deadline arrived with no fds ready.
            return Ok(0);
        }

        let mut wake_fired = false;
        let mut inotify_data = false;
        for ev in &epoll_events[..n_ready] {
            match ev.u64 {
                INOTIFY_TOKEN => inotify_data = true,
                WAKE_TOKEN => wake_fired = true,
                other => tracing::warn!(
                    token = format_args!("{other:#018x}"),
                    "epoll_wait returned unrecognised token (structural break)"
                ),
            }
        }

        if wake_fired {
            // Drain the eventfd counter to clear `EPOLLIN` on the wake
            // fd. The actual counter value is observationally
            // irrelevant — any non-zero accumulation collapses to
            // "wake delivered." The error path is reachable only on a
            // structural break (the watcher's `Arc<OwnedFd>` keeps
            // `wake_fd` alive for the watcher's lifetime); log at
            // trace and proceed.
            if let Err(e) = ffi::eventfd_drain(&self.wake_fd) {
                tracing::trace!(error = ?e, "inotify wake-fd drain failed (benign)");
            }
        }

        if !inotify_data {
            // Wake-only return path. The bin's loop re-checks pending
            // `WatchOp`s + shutdown flag before the next `poll_until`.
            return Ok(0);
        }

        // Read pending records into the pre-allocated drain buffer.
        // `read_inotify` returns `Ok(0)` on `EAGAIN` (queue drained
        // between `epoll_wait` and `read` — impossible under the
        // single-reader watcher discipline; defended).
        let n_bytes = ffi::read_inotify(&self.inotify_fd, &mut self.read_buf)
            .map_err(|e| WatchFailure::from_io(&e))?;
        if n_bytes == 0 {
            return Ok(0);
        }

        let mut emitted = 0usize;
        for rec in record::parse(&self.read_buf[..n_bytes]) {
            // 1. IN_Q_OVERFLOW: queue-wide kernel-side overflow signal.
            if rec.mask & libc::IN_Q_OVERFLOW != 0 {
                out.push(WatcherEvent::Overflow {
                    scope: OverflowScope::Global,
                });
                emitted += 1;
                continue;
            }

            // 2. IN_IGNORED: cleanup signal for this wd.
            if rec.mask & libc::IN_IGNORED != 0 {
                let was_draining = self.draining_wds.remove(&rec.wd);
                if !was_draining && let Some(r) = self.by_wd.remove(&rec.wd) {
                    // Spontaneous reap: the kernel destroyed the watch
                    // because the watched inode was deleted/unmounted
                    // (the preceding IN_DELETE_SELF / IN_UNMOUNT
                    // already emitted Removed / Revoked to the engine).
                    // Clear the entry so a subsequent `unwatch(r)` from
                    // the engine's tear-down sees the stale-id branch
                    // and short-circuits, and so a future kernel-side
                    // wd reuse cannot mis-attribute events through a
                    // stale `by_wd[wd]` mapping.
                    self.by_resource.remove(r);
                }
                continue;
            }

            // 3. Stale event on a draining wd: the rm_watch was issued
            //    but IN_IGNORED hasn't arrived yet. Pre-rm events on
            //    this wd belong to the prior inode; drop. Engine's
            //    reconcile-on-next-probe corrects any state drift.
            if self.draining_wds.contains(&rec.wd) {
                continue;
            }

            // 4. Normal event. Resolve wd → ResourceId.
            let Some(&r) = self.by_wd.get(&rec.wd) else {
                // Defensive guard against routing-table desync. Case 2
                // clears `by_wd` synchronously on every `IN_IGNORED`,
                // and `watch_inner` populates it on every successful
                // `inotify_add_watch`; a surviving miss means the
                // kernel handed us a wd we never registered — a
                // kernel quirk. Drop the event in release; the
                // debug_assert is the CI tripwire if this ever fires
                // under tests.
                debug_assert!(
                    false,
                    "inotify event on unmapped wd ({}); routing-table desync",
                    rec.wd
                );
                tracing::trace!(
                    wd = rec.wd,
                    mask = format_args!("{:#x}", rec.mask),
                    "inotify event on unmapped wd; dropping"
                );
                continue;
            };

            // Cached-kind miss falls back to Unknown — consistent
            // with the kqueue branch's defensive routing for events
            // on a resource whose entry was cleared between the
            // kernel's queue-add and our drain (e.g., a spontaneous
            // IN_IGNORED arm earlier in the same batch).
            let kind = self
                .by_resource
                .get(r)
                .map_or(ResourceKind::Unknown, |e| e.kind);
            let Some(fs_event) = normalize::mask_to_fs_event(rec.mask, kind) else {
                // No actionable bit — registration ack with only
                // orientation flags (IN_ISDIR), or a defensive
                // IN_IGNORED slip the case-2 branch should have
                // caught. Drop silently; not a routing fault.
                continue;
            };

            // Per-drain dedup over `(ResourceId, FsEvent)`. First-write
            // wins, so the kernel's `IN_MODIFY` precedes its
            // `IN_CLOSE_WRITE` in the emitted stream — matching the
            // natural FIFO drain order. Linear scan rather than a
            // sorted set; see [`Self::seen`] for the sizing rationale.
            let key = (r, fs_event);
            if self.seen.contains(&key) {
                continue;
            }
            self.seen.push(key);

            out.push(WatcherEvent::Fs {
                resource: r,
                event: fs_event,
            });
            emitted += 1;
        }

        tracing::trace!(emitted, wake_fired, n_bytes, "inotify drained");
        Ok(emitted)
    }
}

/// Compute the kernel-side `inotify_add_watch` mask, including the
/// install-time directional flags.
///
/// Wraps [`translate::class_set_to_mask`] (the pure event-mask
/// translation: identity floor + defensive flags + per-class bits) and
/// ORs `IN_ONLYDIR` for Dir watches. `IN_ONLYDIR` is documented as a
/// directional flag rather than an event bit (per `inotify(7)`); it
/// belongs in the watcher because it lives outside the
/// [`crate::ClassSet`] vocabulary.
///
/// `IN_ONLYDIR` is defense-in-depth under the watcher's
/// `O_PATH | /proc/self/fd/N` install — the fd already binds to a
/// specific inode and the fstat verification has already confirmed
/// Dir-ness — but the kernel-side guard catches any race the userspace
/// chain missed and is free at install time.
#[must_use]
const fn compute_install_mask(events: ClassSet, kind: ResourceKind) -> u32 {
    let mut mask = translate::class_set_to_mask(events, kind);
    if matches!(kind.effective(), ResourceKind::Dir) {
        mask |= libc::IN_ONLYDIR;
    }
    mask
}
