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

// Several state fields and the `InotifyEntry` type are populated by the
// `FsWatcher` mutators landing in B6 (`watch`), B7 (`unwatch`), and B8
// (`suppress`/`unsuppress`); the drain buffer and `draining_wds` are
// consumed by B9 (`poll_until`). The structure is loaded here so each
// follow-up commit is a focused mutator-only diff. Remove this allow
// once Phase B9 wires every field into the hot path. `derive(Debug)`
// already reads every field, so the struct itself stays warning-free —
// the allow scopes the unused private constants (`INOTIFY_TOKEN`,
// `WAKE_TOKEN`, `READ_BUF_BYTES` consumers in B9) and the
// `InotifyEntry` field access path (B6 first writer).
#![allow(dead_code)]

use crate::inotify::ffi;
use crate::inotify::wake::InotifyWakeHandle;
use crate::{FsWatcher, WakeHandle, WatchFailure, WatcherEvent};
use slotmap::SecondaryMap;
use specter_core::{ClassSet, ResourceId, ResourceKind};
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::os::fd::OwnedFd;
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
}

impl FsWatcher for InotifyWatcher {
    /// Stub. Phase B6 lands the real body — race-free `O_PATH` +
    /// `/proc/self/fd/N` install discipline, mask-cache short-circuit,
    /// and the hardlink-aliasing guard on `wd` collision.
    ///
    /// Returns [`WatchFailure::Invariant`] (errno `ENOSYS`) so the
    /// engine's classification path is explicit if a test fixture
    /// wires a post-`new()`-success `WatchOp::Watch` before B6 ships;
    /// the bin's Linux dispatch (Phase B10) keeps real `WatchOp`s off
    /// this stub on a healthy build.
    fn watch(
        &mut self,
        _r: ResourceId,
        _path: &Path,
        _kind: ResourceKind,
        _events: ClassSet,
    ) -> Result<(), WatchFailure> {
        Err(WatchFailure::Invariant {
            errno: libc::ENOSYS,
        })
    }

    /// Stub. Phase B7 lands the real body — `inotify_rm_watch` plus
    /// the `draining_wds` mark-and-drop that closes the wd-reuse race
    /// window between `rm_watch` and the kernel's `IN_IGNORED`.
    fn unwatch(&mut self, _r: ResourceId) {}

    /// Stub. Phase B8 lands the real body — user-space filter via the
    /// `suppressed` map. inotify has no kernel-level disable analogue,
    /// so the contract names "silenced delivery" rather than
    /// "registration disabled".
    fn suppress(&mut self, _r: ResourceId) {}

    /// Stub. Phase B8 lands the real body. Mirror of `suppress`.
    fn unsuppress(&mut self, _r: ResourceId) {}

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
