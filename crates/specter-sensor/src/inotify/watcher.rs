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
//! re-`watch()` (Phase B6) with an unchanged mask short-circuits
//! without a syscall — the kernel's "replace mask" semantics on an
//! existing path produce the same bits, so the call is a noop. Mirrors
//! kqueue's `registered_fflags` discipline.

// `suppressed` is populated and read by B8 (`suppress`/`unsuppress`);
// `draining_wds` consumption lives in B9 (`poll_until`). `derive(Debug)`
// already reads every field, so this scopes the unused private items
// — primarily the [`InotifyEntry::wd`] reader path B9 wires into the
// `wd → ResourceId` route. Remove once Phase B9 lands.
#![allow(dead_code)]

use crate::inotify::wake::InotifyWakeHandle;
use crate::inotify::{ffi, translate};
use crate::{FsWatcher, WakeHandle, WatchFailure, WatchFailureExt, WatcherEvent};
use slotmap::SecondaryMap;
use specter_core::{ClassSet, ResourceId, ResourceKind};
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Token tagging the inotify fd in epoll. Phase B9's `poll_until`
/// consumer reads `epoll_event.u64` to discriminate inotify-data-ready
/// from wake-fired; distinct from [`WAKE_TOKEN`].
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
    /// only `poll_until` (Phase B9) reads from it; wake handles never
    /// touch it.
    epoll_fd: OwnedFd,

    /// `ResourceId → (wd, mask)`. Populated by `watch()` (Phase B6) on
    /// successful install, cleared by `unwatch()` (Phase B7). The mask
    /// cache lets a re-`watch()` skip the syscall when the install
    /// mask is unchanged (mirror of kqueue's `registered_fflags`).
    by_resource: SecondaryMap<ResourceId, InotifyEntry>,

    /// `wd → ResourceId`. inotify events don't carry userdata
    /// (kqueue's `udata` analogue), so the watcher pays the storage to
    /// route a record's `wd` back to the slot it belongs to. wd values
    /// are dense small integers; `BTreeMap` O(log n) lookups are fine
    /// at typical watch counts and avoid the `HashMap` ban from
    /// `deny.toml` for sensor-side state.
    by_wd: BTreeMap<libc::c_int, ResourceId>,

    /// Per-resource kind cache. Populated at fresh-watch time from the
    /// `fstat` of the freshly opened fd (Phase B6) — closing the
    /// TOCTOU window between the engine's `WatchOp::Watch.kind` and the
    /// kernel's path-resolution at install time. Used by
    /// [`crate::inotify::normalize::mask_to_fs_event`] to disambiguate
    /// `IN_MODIFY` on Dir vs File defensive paths.
    kinds: SecondaryMap<ResourceId, ResourceKind>,

    /// Suppressed set. inotify has no kernel-level disable analogue to
    /// kqueue's `EV_DISABLE`, so `poll_until` (Phase B9) filters
    /// delivery user-side using this map. Mutated by
    /// `suppress`/`unsuppress` (Phase B8); the trait's idempotency
    /// contract is satisfied by `BTreeMap`'s insert/remove semantics.
    suppressed: SecondaryMap<ResourceId, ()>,

    /// `wd`s in the "draining" state: `inotify_rm_watch` has been
    /// called but the kernel's `IN_IGNORED` for that wd has not yet
    /// arrived in our read buffer. Events on draining wds are dropped
    /// during `poll_until` (Phase B9); the `IN_IGNORED` consumption
    /// reaps the flag. See § 1.3 of the inotify port plan for the
    /// wd-reuse race this closes — a subsequent `inotify_add_watch`
    /// may return the same wd before userspace observes the
    /// `IN_IGNORED`, and pre-rm events on the old inode would
    /// otherwise mis-attribute to the freshly attached resource.
    draining_wds: BTreeSet<libc::c_int>,

    /// Drain buffer for inotify event records. Sized at construction
    /// and reused across drains — `poll_until` (Phase B9) performs no
    /// allocation on the hot path.
    read_buf: Vec<u8>,
}

/// Per-resource cached install state.
///
/// `mask` is the exact bits passed to `inotify_add_watch` (including
/// install-time directional flags like `IN_ONLYDIR` for Dir watches).
/// A re-`watch()` (Phase B6) recomputes the mask from the user's
/// `events` set and the cached kind; an unchanged mask short-circuits
/// without a syscall — the kernel's "replace mask" semantics on an
/// existing path would produce identical bits.
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
    ///   refcount (R2 / D11). The cached mask short-circuits when
    ///   unchanged; an inode-swap is detected via the `wd != prior.wd`
    ///   check (atomic rename swapped the path between the prior
    ///   install and this re-add) and the prior wd is drained.
    ///
    /// - **Fresh-watch** — `r` has no entry. Triggered on the 0→1
    ///   `watch_demand` edge. Race-free install via
    ///   [`ffi::open_o_path`] + `/proc/self/fd/N` (§ 1.2 of the
    ///   inotify port plan): the fd binds to a specific inode, and
    ///   `inotify_add_watch` on the magic-symlink path resolves to
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
    /// resource's kernel-side registration entirely (§ 1.7 of the
    /// plan). The restoration is best-effort; on failure the existing
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
    /// from the next `poll_until` (Phase B9) iteration; the kernel's
    /// synchronous `IN_IGNORED` arrives later in the drain stream and
    /// reaps the flag. See § 1.3 of the inotify port plan.
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

    /// Silence event delivery on `r` via a user-space filter — inotify
    /// has no kernel-level disable analogue to kqueue's `EV_DISABLE`,
    /// so the [`Self::suppressed`] map is the authoritative source and
    /// `poll_until` (Phase B9) consults it before lifting an event
    /// onto the engine's input channel.
    ///
    /// Idempotent on stale ids (the engine emits Suppress only on the
    /// 0→1 `suppress_count` edge under D11, but the operation is
    /// safe under any race). The trait doc tightening (Phase A6)
    /// pins the kernel-disable vs user-space-filter wording so the
    /// engine's caller doesn't depend on a kernel-side flush
    /// semantic kqueue happens to provide and inotify cannot.
    ///
    /// Mirrors the kqueue branch's "warn on unwatched" discipline so
    /// the two backends produce comparable diagnostic output on a
    /// `WatchOp::Suppress` that races a concurrent `WatchOp::Unwatch`.
    fn suppress(&mut self, r: ResourceId) {
        if !self.by_resource.contains_key(r) {
            tracing::warn!(?r, "inotify suppress on unwatched resource (race; dropped)");
            return;
        }
        self.suppressed.insert(r, ());
        tracing::debug!(?r, "inotify suppress (user-space)");
    }

    /// Restore event delivery on `r`. Idempotent; mirror of
    /// [`Self::suppress`] including the "warn on unwatched" race
    /// discipline. A watched-but-not-suppressed `r` is silently fine
    /// — the [`Self::suppressed`] removal is a no-op and event
    /// delivery resumes (or, more accurately, was already happening).
    /// Symmetry with kqueue's
    /// [`crate::kqueue::watcher::KqueueWatcher::unsuppress`] keeps the
    /// engine's caller backend-agnostic; the kqueue branch's
    /// `EV_ENABLE` syscall is the structural counterpart of this
    /// `suppressed.remove(r)` call.
    fn unsuppress(&mut self, r: ResourceId) {
        if !self.by_resource.contains_key(r) {
            tracing::warn!(
                ?r,
                "inotify unsuppress on unwatched resource (race; dropped)"
            );
            return;
        }
        self.suppressed.remove(r);
        tracing::debug!(?r, "inotify unsuppress (user-space)");
    }

    /// Stub. Phase B9 lands the real body — `epoll_wait` over
    /// `(inotify_fd, wake_fd)` with `IN_IGNORED` consumption,
    /// `IN_Q_OVERFLOW` lifting to [`WatcherEvent::Overflow`], and
    /// per-batch dedup (inotify, unlike kqueue's `EV_CLEAR`, doesn't
    /// coalesce kernel-side).
    ///
    /// Returns `Ok(0)` ("no events available") rather than an error
    /// so a caller that bypasses the bin's Linux dispatch gating
    /// during the helper-cluster phases doesn't see a misleading hard
    /// failure on the watcher thread's drain loop.
    fn poll_until(
        &mut self,
        _deadline: Option<Instant>,
        _out: &mut Vec<WatcherEvent>,
    ) -> Result<usize, WatchFailure> {
        Ok(0)
    }

    /// Capture a wake handle. Real from B5: clones the watcher's
    /// `Arc<OwnedFd>` of the eventfd so the handle survives the
    /// watcher's drop without UB (see [`InotifyWakeHandle`] for the
    /// `Arc` discipline). Cheap (one `Arc` increment + one `Box`
    /// allocation); idempotent — multiple handles coexist freely.
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
