//! `InotifyWatcher` — inotify-backed `FsWatcher` impl.
//!
//! Single-threaded: one thread owns the [`InotifyWatcher`] value and
//! drives [`FsWatcher::watch`] / [`FsWatcher::unwatch`] /
//! [`FsWatcher::suppress`] / [`FsWatcher::unsuppress`] between
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
//! Each entry caches `(wd, mask)`: the watch descriptor returned by
//! `inotify_add_watch` and the kernel-side mask we last installed. A
//! re-`watch()` with an unchanged mask short-circuits without a
//! syscall — the kernel's "replace mask" semantics on an existing path
//! produce the same bits, so the call is a noop. Mirrors kqueue's
//! `registered_fflags` discipline.

use crate::inotify::wake::InotifyWakeHandle;
use crate::inotify::{ffi, normalize, record, translate};
use crate::{FsWatcher, WakeHandle, WatchFailure, WatchFailureExt, WatcherEvent};
use slotmap::SecondaryMap;
use specter_core::{ClassSet, FsEvent, OverflowScope, ResourceId, ResourceKind};
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};
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

    /// `ResourceId → (wd, mask)`. Populated by `watch()` on successful
    /// install, cleared by `unwatch()`. The mask cache lets a
    /// re-`watch()` skip the syscall when the install mask is unchanged
    /// (mirror of kqueue's `registered_fflags`).
    by_resource: SecondaryMap<ResourceId, InotifyEntry>,

    /// `wd → ResourceId`. inotify events don't carry userdata
    /// (kqueue's `udata` analogue), so the watcher pays the storage to
    /// route a record's `wd` back to the slot it belongs to. wd values
    /// are dense small integers; `BTreeMap` O(log n) lookups are fine
    /// at typical watch counts and avoid the `HashMap` ban from
    /// `deny.toml` for sensor-side state.
    by_wd: BTreeMap<libc::c_int, ResourceId>,

    /// Per-resource kind cache. Populated at fresh-watch time from the
    /// `fstat` of the freshly opened fd — closing the TOCTOU window
    /// between the engine's `WatchOp::Watch.kind` and the
    /// kernel's path-resolution at install time. Used by
    /// [`crate::inotify::normalize::mask_to_fs_event`] to disambiguate
    /// `IN_MODIFY` on Dir vs File defensive paths.
    kinds: SecondaryMap<ResourceId, ResourceKind>,

    /// Suppressed set. `poll_until` consults this map before lifting an
    /// event onto the engine's input channel; events for a suppressed
    /// resource drop at the watcher boundary per the
    /// [`FsWatcher::suppress`] contract. Mutated by `suppress` /
    /// `unsuppress`; idempotency is satisfied by `SecondaryMap`'s
    /// insert/remove semantics.
    suppressed: SecondaryMap<ResourceId, ()>,

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
    /// and reused across drains — `poll_until` performs no allocation
    /// on the hot path.
    read_buf: Vec<u8>,
}

/// Per-resource cached install state.
///
/// `mask` is the exact bits passed to `inotify_add_watch` (including
/// install-time directional flags like `IN_ONLYDIR` for Dir watches).
/// A re-`watch()` recomputes the mask from the user's `events` set and
/// the cached kind; an unchanged mask short-circuits without a syscall
/// — the kernel's "replace mask" semantics on an existing path would
/// produce identical bits.
#[derive(Debug, Clone, Copy)]
struct InotifyEntry {
    wd: libc::c_int,
    mask: u32,
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
            kinds: SecondaryMap::new(),
            suppressed: SecondaryMap::new(),
            draining_wds: BTreeSet::new(),
            read_buf: vec![0u8; READ_BUF_BYTES],
        })
    }

    /// Internal `watch` body returning the raw `io::Error` set; the
    /// trait wrapper maps that into a typed [`WatchFailure`] at the
    /// boundary so `?` propagation across the open / fstat / add_watch
    /// chain stays uniform.
    ///
    /// # Branches
    ///
    /// - **Re-watch** — `r` already holds an entry. Triggered by the
    ///   engine when `Resource.events_union` changes at non-zero
    ///   refcount. The cached mask short-circuits when unchanged; an
    ///   inode-swap is detected via the `wd != prior.wd`
    ///   check (atomic rename swapped the path between the prior
    ///   install and this re-add) and the prior wd is drained.
    ///
    /// - **Fresh-watch** — `r` has no entry. Triggered on the 0→1
    ///   `watch_demand` edge. Race-free install via
    ///   [`ffi::open_o_path`] + `/proc/self/fd/N`: the fd binds to a
    ///   specific inode, and `inotify_add_watch` on the magic-symlink
    ///   path resolves to
    ///   that inode regardless of intervening renames at `path`. The
    ///   fstat verification then matches the engine's expected
    ///   `kind`; a kind disagreement maps to `ENOTDIR`, which the
    ///   trait wrapper classifies as [`WatchFailure::Resource`] so
    ///   the engine routes through the path-fatal recovery channel.
    ///
    /// # Hardlink aliasing
    ///
    /// Two `ResourceId`s pointing to the same inode receive the same
    /// `wd` from the kernel — there is one kernel-side watch entry
    /// per `(inotify_fd, inode)` pair (per `inotify(7)`'s "the
    /// existing watch is updated" semantics). v1 rejects the second
    /// attachment but the rejection branch *restores* the existing
    /// resource's mask via a follow-up `inotify_add_watch` on the
    /// same `/proc/self/fd/N`: the kernel's "replace mask" semantics
    /// have just clobbered the existing watch with our new mask, and
    /// a naive `inotify_rm_watch` would tear down the existing
    /// resource's kernel-side registration entirely. The restoration
    /// is best-effort; on failure the existing
    /// resource's mask remains the rejected resource's mask until its
    /// next reconcile triggers a re-add — a documented v1 limitation,
    /// not a correctness regression.
    fn watch_inner(
        &mut self,
        r: ResourceId,
        path: &Path,
        kind: ResourceKind,
        events: ClassSet,
    ) -> io::Result<()> {
        // ── Re-watch path ───────────────────────────────────────────
        if let Some(prior) = self.by_resource.get(r).copied() {
            let cached_kind = self.kinds.get(r).copied().unwrap_or(ResourceKind::Unknown);
            let new_mask = compute_install_mask(events, cached_kind);
            if new_mask == prior.mask {
                tracing::trace!(
                    ?r,
                    ?events,
                    mask = format_args!("{new_mask:#x}"),
                    "inotify re-watch noop (mask unchanged)"
                );
                return Ok(());
            }

            // Re-open + fstat. The cached_kind is the engine's
            // authoritative classification for `r`; verify the inode
            // shape hasn't mutated under the path. A disagreement
            // maps to `ENOTDIR` (path-fatal) so the engine reseeds
            // via descent rather than installing a kind-incoherent
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

            let proc_path = format!("/proc/self/fd/{}", fd.as_raw_fd());
            let proc_path_ref = Path::new(&proc_path);
            let wd = ffi::inotify_add_watch(&self.inotify_fd, proc_path_ref, new_mask)?;

            // Inode-swap detection. A different wd means the path now
            // resolves to a different inode (atomic rename swapped
            // the path between the prior install and this re-add).
            // Mark the prior wd as draining so any pre-rm events on
            // it are dropped from the next `poll_until` iteration —
            // the kernel's `IN_IGNORED` arrives later in the drain
            // stream and reaps the flag.
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
                    // arrive on the drain stream; `draining_wds`
                    // covers the gap.
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

            // Hardlink aliasing guard.
            if let Some(&existing) = self.by_wd.get(&wd)
                && existing != r
            {
                self.reject_aliased_install(existing, r, wd, proc_path_ref);
                return Err(io::Error::from_raw_os_error(libc::EEXIST));
            }

            self.by_resource
                .insert(r, InotifyEntry { wd, mask: new_mask });
            self.by_wd.insert(wd, r);
            tracing::debug!(
                ?r,
                wd,
                old_mask = format_args!("{:#x}", prior.mask),
                new_mask = format_args!("{new_mask:#x}"),
                "inotify rewatch (mask changed)"
            );
            return Ok(());
        }

        // ── Fresh-watch path ────────────────────────────────────────
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
        let mask = compute_install_mask(events, observed_kind);

        // 5) Install via `/proc/self/fd/N`. The kernel's procfs
        //    resolver returns the exact inode the fd refers to,
        //    closing the TOCTOU window between fstat and add_watch
        //    that a naive `inotify_add_watch(path)` would leave open.
        let proc_path = format!("/proc/self/fd/{}", fd.as_raw_fd());
        let proc_path_ref = Path::new(&proc_path);
        let wd = ffi::inotify_add_watch(&self.inotify_fd, proc_path_ref, mask)?;

        // 6) Hardlink aliasing guard. `fd` is intentionally still
        //    alive here so `reject_aliased_install` can re-use the
        //    same `/proc/self/fd/N` for the kernel-side mask
        //    restoration on the existing resource's behalf.
        if let Some(&existing) = self.by_wd.get(&wd)
            && existing != r
        {
            self.reject_aliased_install(existing, r, wd, proc_path_ref);
            return Err(io::Error::from_raw_os_error(libc::EEXIST));
        }

        // 7) Commit. `fd` drops at end of scope; the kernel watches
        //    the inode the fd resolved to at `add_watch` time,
        //    independent of fd lifetime (per `inotify(7)`).
        self.by_resource.insert(r, InotifyEntry { wd, mask });
        self.by_wd.insert(wd, r);
        self.kinds.insert(r, observed_kind);
        tracing::debug!(
            ?r,
            ?path,
            kind = ?observed_kind,
            ?events,
            wd,
            mask = format_args!("{mask:#x}"),
            "inotify watch"
        );
        Ok(())
    }

    /// Common epilogue for the two hardlink-aliasing rejection sites.
    ///
    /// Restores `existing`'s kernel-side mask via a follow-up
    /// `inotify_add_watch` on the same `/proc/self/fd/N`, then clears
    /// every map keyed by `r` so the engine sees a clean
    /// "unwatched-and-unknown" state on retry. The two re-add and
    /// remove steps are separate so the borrow checker tolerates the
    /// intermediate `&self.by_resource` immutable borrow before the
    /// `self.by_resource.remove(r)` mutable borrow.
    ///
    /// `proc_path_ref` references a `String` owned by the caller; the
    /// caller must keep the underlying [`OwnedFd`] alive across this
    /// call (the kernel resolves the magic-symlink path at syscall
    /// time, and a closed fd would surface as `ENOENT`).
    fn reject_aliased_install(
        &mut self,
        existing: ResourceId,
        r: ResourceId,
        wd: libc::c_int,
        proc_path_ref: &Path,
    ) {
        match self.by_resource.get(existing).copied() {
            Some(existing_entry) => {
                if let Err(e) =
                    ffi::inotify_add_watch(&self.inotify_fd, proc_path_ref, existing_entry.mask)
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
        // to r — remove every entry so the engine's clamp →
        // re-resolve flow starts from a clean slate.
        self.by_resource.remove(r);
        self.kinds.remove(r);
        self.suppressed.remove(r);
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
    /// Idempotent on stale ids — clearing the side maps (`suppressed`,
    /// `kinds`) before the `by_resource` removal is safe regardless of
    /// whether `r` was actually held; the kernel-side `rm_watch` only
    /// runs when `by_resource` had an entry to remove.
    ///
    /// `EINVAL` from `rm_watch` is benign: the kernel had already
    /// reaped the wd (the inode was deleted out from under us) and
    /// queued `IN_IGNORED` synchronously at that time. The
    /// `draining_wds` flag still covers the window — any pre-deletion
    /// events queued before the kernel's reap are dropped, and the
    /// `IN_IGNORED` consumption clears the flag.
    fn unwatch(&mut self, r: ResourceId) {
        self.suppressed.remove(r);
        self.kinds.remove(r);
        let Some(entry) = self.by_resource.remove(r) else {
            // Stale id — every map keyed by `r` is now empty (or was
            // already empty); no kernel-side work to do.
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

    /// Silence event delivery on `r`. Inserts into [`Self::suppressed`];
    /// `poll_until` consults the map before lifting an event onto the
    /// engine's input channel, so events for a suppressed resource drop
    /// at the watcher boundary per the [`FsWatcher::suppress`] contract.
    ///
    /// Idempotent on stale ids (the engine emits Suppress only on the
    /// 0→1 `suppress_count` edge, but the operation is safe under any
    /// race). Warns on an unwatched id so a `WatchOp::Suppress` that
    /// races a concurrent `WatchOp::Unwatch` is observable in operator
    /// logs — mirrored by the kqueue branch.
    fn suppress(&mut self, r: ResourceId) {
        if !self.by_resource.contains_key(r) {
            tracing::warn!(?r, "inotify suppress on unwatched resource (race; dropped)");
            return;
        }
        self.suppressed.insert(r, ());
        tracing::debug!(?r, "inotify suppress");
    }

    /// Restore event delivery on `r`. Idempotent; mirror of
    /// [`Self::suppress`] including the "warn on unwatched" race
    /// discipline. A watched-but-not-suppressed `r` is silently fine
    /// — the [`Self::suppressed`] removal is a no-op and event
    /// delivery resumes (or, more accurately, was already happening).
    fn unsuppress(&mut self, r: ResourceId) {
        if !self.by_resource.contains_key(r) {
            tracing::warn!(
                ?r,
                "inotify unsuppress on unwatched resource (race; dropped)"
            );
            return;
        }
        self.suppressed.remove(r);
        tracing::debug!(?r, "inotify unsuppress");
    }

    /// Block on `epoll_wait` over `(inotify_fd, wake_fd)` until the
    /// deadline, an inotify record, or a wake fires. Drains records
    /// into `out` and returns the number of [`WatcherEvent`]s pushed
    /// this call. Mirror of [`crate::kqueue::watcher::KqueueWatcher::poll_until`]
    /// over the inotify substrate.
    ///
    /// # Record dispatch
    ///
    /// Per-record demux into four cases:
    ///
    /// 1. **`IN_Q_OVERFLOW`** — kernel signal that the per-instance
    ///    event queue overflowed and dropped record(s). Lift to
    ///    [`WatcherEvent::Overflow`] with [`OverflowScope::Global`];
    ///    the engine reseeds every in-scope Profile.
    ///    Per `inotify(7)` the record's `wd` is `-1` and no other
    ///    actionable bit is set; the by_wd / draining checks below
    ///    are bypassed via the early `continue`.
    /// 2. **`IN_IGNORED`** — kernel cleanup signal; the watch
    ///    descriptor is being reaped. Two legitimate paths reach
    ///    here, distinguished by `draining_wds` membership: a
    ///    watcher-initiated `inotify_rm_watch` (the wd is in
    ///    `draining_wds` and per-resource state was already cleared
    ///    by [`Self::unwatch`] — only the flag needs clearing here),
    ///    or a kernel-side spontaneous reap (the watched inode was
    ///    deleted/unmounted; the watcher still holds the per-resource
    ///    maps and clears them now so the engine's eventual
    ///    `Unwatch` finds a clean slate).
    /// 3. **Stale event on a draining wd** — the kernel queued events
    ///    on the wd before our `inotify_rm_watch`, but `IN_IGNORED`
    ///    hasn't yet been consumed. Drop the event: the inode the wd
    ///    referred to is no longer the ResourceId's intent, and a
    ///    subsequent kernel-side wd reuse (a fresh `inotify_add_watch`
    ///    receiving the same wd) would otherwise mis-attribute pre-rm
    ///    events to the freshly attached resource.
    /// 4. **Normal event** — resolve `wd → ResourceId` via `by_wd`,
    ///    drop if [`Self::suppressed`] holds the resource (user-space
    ///    filter; inotify has no kernel-level disable), normalize via
    ///    [`normalize::mask_to_fs_event`] using the cached kind, and
    ///    push as [`WatcherEvent::Fs`].
    ///
    /// # Per-batch dedup
    ///
    /// inotify (unlike kqueue's `EV_CLEAR`) does not coalesce
    /// kernel-side: a single write produces both `IN_MODIFY` and
    /// `IN_CLOSE_WRITE` records, which both normalize to
    /// [`FsEvent::Modified`] for the same resource. A
    /// `BTreeSet<(ResourceId, FsEvent)>` collapses identical pairs
    /// within this drain. Per-batch only — across-batch dedup would
    /// suppress the engine's `Settling` debounce signal (the engine
    /// debounces by re-arming its settle timer on every event;
    /// suppressing redundant events across batches would let a
    /// long-running write stream go undebounced after the first batch).
    ///
    /// # Wake handling
    ///
    /// `epoll_wait` distinguishes wake-fired from inotify-data-ready
    /// via the per-fd token (`WAKE_TOKEN` vs `INOTIFY_TOKEN`). A wake
    /// alone — without inotify data — drains the eventfd counter and
    /// returns `Ok(0)` so the bin's drain loop re-checks pending
    /// `WatchOp`s and shutdown flag. Concurrent wakes accumulate in
    /// the eventfd counter; one drain consumes them all atomically.
    ///
    /// # Errors
    ///
    /// Syscall failures classify through [`WatchFailureExt::from_io`]
    /// at the trait boundary — symmetric with [`Self::watch`] / kqueue.
    /// `EINTR` is retried inside the FFI helpers (see
    /// [`ffi::epoll_wait`] / [`ffi::read_inotify`]); the bin treats a
    /// non-`EINTR` failure as terminal for the watcher thread.
    fn poll_until(
        &mut self,
        deadline: Option<Instant>,
        out: &mut Vec<WatcherEvent>,
    ) -> Result<usize, WatchFailure> {
        // `None` blocks indefinitely (-1); `Some(d)` past the deadline
        // saturates to `Duration::ZERO` ⇒ `0 ms` non-blocking poll.
        let timeout_ms = deadline.map_or(-1, |d| {
            ffi::duration_to_ms(d.saturating_duration_since(Instant::now()))
        });

        // Two slots — one per epoll-registered fd. Both can be ready
        // at once (deadline + a concurrent wake + inotify data).
        let mut epoll_events = [libc::epoll_event { events: 0, u64: 0 }; 2];
        let n_ready = ffi::epoll_wait(&self.epoll_fd, &mut epoll_events, timeout_ms)
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
        // Per-batch dedup over `(ResourceId, FsEvent)`. First-write
        // wins (`BTreeSet::insert` returns `false` on duplicates), so
        // the kernel's `IN_MODIFY` precedes its `IN_CLOSE_WRITE` in the
        // emitted stream — matching the natural FIFO drain order.
        let mut seen: BTreeSet<(ResourceId, FsEvent)> = BTreeSet::new();

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
                    // Clear per-resource state so a subsequent
                    // `unwatch(r)` from the engine's tear-down sees the
                    // stale-id branch and short-circuits, and so a
                    // future kernel-side wd reuse cannot mis-attribute
                    // events through a stale `by_wd[wd]` mapping.
                    self.by_resource.remove(r);
                    self.kinds.remove(r);
                    self.suppressed.remove(r);
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
                // Unmapped wd reaching this branch is structurally
                // unreachable: case 2 cleared by_wd synchronously on
                // every IN_IGNORED, and `watch_inner` populates by_wd
                // on every successful `inotify_add_watch`. A surviving
                // miss means the kernel handed us a wd we never
                // registered — a kernel quirk; log at trace and drop.
                tracing::trace!(
                    wd = rec.wd,
                    mask = format_args!("{:#x}", rec.mask),
                    "inotify event on unmapped wd; dropping"
                );
                continue;
            };

            // User-space suppression filter — drop events for resources
            // the engine has marked suppressed. Symmetric with the
            // kqueue branch's same gate; satisfies the
            // [`FsWatcher::suppress`] drop-at-boundary contract.
            if self.suppressed.contains_key(r) {
                continue;
            }

            // Cached-kind miss falls back to Unknown — consistent with
            // the kqueue branch's defensive routing for events on a
            // resource whose `kinds` slot was cleared between the
            // kernel's queue-add and our drain (e.g., race against a
            // spontaneous IN_IGNORED in the same batch).
            let kind = self.kinds.get(r).copied().unwrap_or(ResourceKind::Unknown);
            let Some(fs_event) = normalize::mask_to_fs_event(rec.mask, kind) else {
                // No actionable bit — registration ack with only
                // orientation flags (IN_ISDIR), or a defensive
                // IN_IGNORED slip the case-2 branch should have
                // caught. Drop silently; not a routing fault.
                continue;
            };

            if !seen.insert((r, fs_event)) {
                continue;
            }

            out.push(WatcherEvent::Fs {
                resource: r,
                event: fs_event,
            });
            emitted += 1;
        }

        tracing::trace!(emitted, n_bytes, "inotify drained");
        Ok(emitted)
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
