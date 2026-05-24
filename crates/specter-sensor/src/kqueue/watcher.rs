//! `KqueueWatcher` — kqueue-backed `FsWatcher` impl.
//!
//! Single-threaded: one thread owns the `KqueueWatcher` value and drives
//! `watch` / `unwatch` between `poll_until` calls. The wake handle
//! ([`KqueueWakeHandle`]) is the only cross-thread surface — see
//! [`crate::kqueue::wake`] for the lifecycle discipline.
//!
//! # Drop semantics
//!
//! Default field-order drop:
//! - `by_resource` drops every [`KqueueEntry`] — the contained
//!   [`OwnedFd`] closes and the kernel removes its vnode registration
//!   on close; the `fflags` and `kind` bookkeeping drops alongside.
//! - `kq` (Arc) decrements; if last, the kqueue fd closes,
//!   kernel-reaping the `EVFILT_USER` ident and any queued events.
//!
//! Wake handles holding Arc clones keep the kqueue fd alive past the
//! watcher's drop — `wake()` from those becomes a no-op-equivalent (no
//! consumer drains the resulting event), with no UB.
//!
//! # Per-resource entry cache
//!
//! Each entry caches `(fd, fflags, kind)`: the watched fd, the
//! post-translation kqueue fflags last installed via `EV_ADD`, and the
//! fstat-verified inode shape from fresh-watch time. The triple is
//! stored as a single struct (not three parallel maps keyed by
//! `ResourceId`) so every install path writes it atomically and every
//! teardown clears it atomically. Mirror of inotify's
//! `InotifyEntry { wd, mask, kind }`.
//!
//! The engine emits `WatchOp::Watch` whenever `Resource.events_union`
//! changes, not only on the 0→1 refcount edge. A re-`watch()` with an
//! unchanged mask skips the syscall entirely; a re-`watch()` with a
//! widened/narrowed mask re-registers via `EV_ADD` (which overwrites
//! the prior fflags) without closing or reopening the fd. The cache
//! is invalidated only on `unwatch` (the engine-side `Unwatch` op,
//! sourced from `sub_watch`'s non-empty → empty edge or
//! `Tree::vacate`'s terminus emission).

use crate::kqueue::wake::KqueueWakeHandle;
use crate::kqueue::{ffi, normalize, translate};
use crate::{FsWatcher, WakeHandle, WatchFailure, WatchFailureExt, WatcherEvent};
use slotmap::{Key, KeyData, SecondaryMap};
use specter_core::{ClassSet, ResourceId, ResourceKind};
use std::io;
use std::os::fd::OwnedFd;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

/// Wake-up ident reserved on the kqueue's `EVFILT_USER` filter. The
/// value is arbitrary — kqueue keys events by `(ident, filter)` and
/// `EVFILT_USER` lives in a different namespace from `EVFILT_VNODE`
/// (where idents are fds), so any non-zero `usize` works. `0xDEAD_BEEF`
/// is a recognizable sentinel in debug output.
const WAKE_IDENT: usize = 0xDEAD_BEEF;

/// Maximum events drained per `kevent` syscall. Excess sit in the
/// kernel queue until the next `poll_until`. 64 mirrors notify-rs and
/// is well above the per-iteration burst the engine produces.
const EVENT_BATCH: usize = 64;

#[derive(Debug)]
pub struct KqueueWatcher {
    /// `ResourceId → (fd, fflags, kind)`. Populated by `watch()` on
    /// successful install, cleared by `unwatch()`. The fflags cache
    /// lets a re-`watch()` skip the syscall when the install mask is
    /// unchanged; the kind cache is consumed by
    /// `normalize::kevent_to_fs_event` for File-vs-Dir disambiguation.
    /// See [`KqueueEntry`] for the field-level lifecycle.
    by_resource: SecondaryMap<ResourceId, KqueueEntry>,
    /// `Arc` so wake handles can hold their own clones without
    /// borrowing from the watcher; drop of the last clone closes the
    /// kqueue fd.
    kq: Arc<OwnedFd>,
}

/// Per-resource cached install state — the `(fd, fflags, kind)` triple
/// installed at fresh-watch time.
///
/// - `fd` is the watched [`OwnedFd`]. The kernel keys the vnode
///   registration off the fd; closing the fd auto-removes the
///   registration. Held for the entry's lifetime — kqueue's re-watch
///   never reopens (the fd is inode-bound, so the cached `kind` cannot
///   become stale relative to the kernel-side registration).
/// - `fflags` is the post-translation kqueue mask last passed to
///   `EV_ADD`. A re-`watch()` recomputes from the user's `events` set
///   and the cached kind; an unchanged mask short-circuits without a
///   syscall — `EV_ADD` is idempotent in mask, so the kernel would
///   produce the same registration.
/// - `kind` is the fstat-verified inode shape at fresh-watch time,
///   closing the TOCTOU window between the engine's
///   `WatchOp::Watch.kind` and the kernel's path-resolution at install
///   time. Invariant for the entry's lifetime — the fd is inode-bound
///   (open() resolves the path once); an inode swap at the path
///   surfaces as `NOTE_DELETE` / `NOTE_RENAME` on the original fd, the
///   engine reseeds via `Unwatch` (which drops the entry; fd close
///   auto-removes the kernel-side registration) followed by a fresh
///   `Watch` against the new inode. Consumed by
///   [`crate::kqueue::normalize::kevent_to_fs_event`] to disambiguate
///   `NOTE_WRITE` / `NOTE_LINK` on Dir vs File, and by
///   `translate::class_set_to_fflags` to compute the per-FD mask.
///
/// Stored as a single struct so the triple is atomic: every install
/// path writes all three fields together, every teardown clears them
/// together. Mirror of inotify's `InotifyEntry { wd, mask, kind }`.
///
/// `OwnedFd` is not `Copy`, so the entry isn't either — callers that
/// need to read `fflags` / `kind` while preserving the fd hold the
/// entry by reference.
#[derive(Debug)]
struct KqueueEntry {
    fd: OwnedFd,
    fflags: u32,
    kind: ResourceKind,
}

impl KqueueWatcher {
    /// Create a fresh kqueue and register the wake-up `EVFILT_USER`
    /// ident. Returns the syscall error on `kqueue()` failure (`EMFILE`,
    /// `ENOMEM` are the only cases — the bin should treat startup
    /// failures as fatal).
    pub fn new() -> io::Result<Self> {
        let kq = Arc::new(ffi::kqueue_new()?);
        ffi::register_user_event(&kq, WAKE_IDENT)?;
        Ok(Self {
            by_resource: SecondaryMap::new(),
            kq,
        })
    }

    /// Internal `watch` body — dispatches by entry presence. The trait
    /// wrapper maps the inner `io::Error` set into a typed
    /// [`WatchFailure`] at the boundary so `?` propagation stays
    /// uniform across the open / stat / register chain.
    ///
    /// - `by_resource[r]` populated → [`Self::rewatch_inner`]
    ///   (re-register via `EV_ADD` on the cached fd; no reopen).
    /// - `by_resource[r]` empty → [`Self::fresh_watch_inner`]
    ///   (open + fstat verify + register).
    ///
    /// The engine-supplied `kind` is structurally irrelevant on
    /// rewatch (the cached kind is the authoritative value; the fd
    /// is inode-bound so the kind cannot become stale relative to
    /// the kernel-side registration). The split makes this explicit
    /// at the call boundary — [`Self::rewatch_inner`] does not take
    /// `kind` or `path` (it reuses the cached fd).
    fn watch_inner(
        &mut self,
        r: ResourceId,
        path: &Path,
        kind: ResourceKind,
        events: ClassSet,
    ) -> io::Result<()> {
        if self.by_resource.contains_key(r) {
            self.rewatch_inner(r, events)
        } else {
            self.fresh_watch_inner(r, path, kind, events)
        }
    }

    /// Re-register `r`'s entry via `EV_ADD` on the cached fd. Engine
    /// triggers this when `Resource.events_union` changes at non-zero
    /// refcount.
    ///
    /// The cached fflags short-circuit when the install mask is
    /// unchanged. On a changed mask, the watcher re-registers via
    /// `EV_ADD` (which overwrites the prior fflags) without closing
    /// or reopening the fd — kqueue's vnode registration is fd-bound,
    /// so the cached kind is invariant and `path` is unused on this
    /// path.
    ///
    /// # Precondition
    ///
    /// `by_resource[r]` must be populated. The dispatcher
    /// [`Self::watch_inner`] enforces this; calling this directly
    /// with an empty entry panics.
    fn rewatch_inner(&mut self, r: ResourceId, events: ClassSet) -> io::Result<()> {
        let prior = self
            .by_resource
            .get(r)
            .expect("rewatch_inner invoked without existing entry");
        let cached_kind = prior.kind;
        let cached_fflags = prior.fflags;
        let install_fflags = translate::class_set_to_fflags(events, cached_kind);
        if install_fflags == cached_fflags {
            tracing::trace!(
                ?r,
                ?events,
                fflags = format_args!("{cached_fflags:#x}"),
                "kqueue re-watch noop (mask unchanged)"
            );
            return Ok(());
        }
        ffi::register_vnode(&self.kq, &prior.fd, r.data().as_ffi(), install_fflags)?;
        // NLL: `prior`'s immutable borrow ends here — its final use
        // was `&prior.fd` above. Now safe to mutate.
        self.by_resource
            .get_mut(r)
            .expect("entry was just observed via `get`")
            .fflags = install_fflags;
        tracing::debug!(
            ?r,
            ?events,
            old_fflags = format_args!("{cached_fflags:#x}"),
            new_fflags = format_args!("{install_fflags:#x}"),
            "kqueue re-register (mask changed)"
        );
        Ok(())
    }

    /// First-install for `r` on the 0→1 `watch_demand` edge.
    ///
    /// Five steps; each failure drops anything earlier (the
    /// [`OwnedFd`] auto-closes) so a partially-failed `watch` leaves
    /// zero state:
    ///
    /// 1. Open the path.
    /// 2. Stat the fd and verify against the engine's expected kind.
    /// 3. Translate to kqueue fflags.
    /// 4. Register via `EV_ADD`.
    /// 5. Insert the entry.
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
        let fd = ffi::open_for_watch(path)?;
        let observed_kind = ffi::stat_kind(&fd)?;
        if !kind.matches_or_unknown(observed_kind) {
            tracing::warn!(
                ?r,
                ?path,
                expected = ?kind,
                observed = ?observed_kind,
                "kqueue watch kind mismatch — engine expected != fstat",
            );
            // ENOTDIR is the canonical "kind disagreement" signal both
            // kqueue and inotify use; the trait wrapper classifies it
            // as `WatchFailure::Resource` so the engine routes through
            // the path-fatal recovery channel.
            return Err(io::Error::from_raw_os_error(libc::ENOTDIR));
        }
        let install_fflags = translate::class_set_to_fflags(events, observed_kind);
        ffi::register_vnode(&self.kq, &fd, r.data().as_ffi(), install_fflags)?;
        // Commit the (fd, fflags, kind) triple atomically. The fd
        // moves into the entry; subsequent `register_vnode` calls on
        // re-watch reuse `&prior.fd`.
        self.by_resource.insert(
            r,
            KqueueEntry {
                fd,
                fflags: install_fflags,
                kind: observed_kind,
            },
        );
        tracing::debug!(
            ?r,
            ?path,
            kind = ?observed_kind,
            ?events,
            fflags = format_args!("{install_fflags:#x}"),
            "kqueue watch"
        );
        Ok(())
    }

    /// One blocking drain of the kqueue: returns when `kevent` reports
    /// activity, the deadline elapses, or a wake fires. Excess events
    /// past [`EVENT_BATCH`] stay queued in the kernel until the next
    /// call.
    ///
    /// Wake events (`EVFILT_USER` carrying [`WAKE_IDENT`]) are
    /// filtered silently before the per-record normalize loop — a
    /// wake-only return surfaces as `Ok(0)`. Real events normalise via
    /// [`normalize::kevent_to_fs_event`] and push as
    /// [`WatcherEvent::Fs`]; kqueue never emits
    /// [`WatcherEvent::Overflow`] (EV_CLEAR coalesces but never silently
    /// drops — overflow is an inotify-only concept).
    ///
    /// `EINTR` retry + per-iteration remaining-budget recompute live
    /// inside [`ffi::kevent_drain`]; the caller's deadline is the
    /// total wall-clock budget.
    ///
    /// `&self` rather than `&mut self`: the watcher's mutable state
    /// (the `(fd, fflags, kind)` per-resource cache) is read here but
    /// only mutated by `watch` / `unwatch`. The kernel queue belongs to
    /// the kq fd, which the watcher holds behind an `Arc`; the syscall
    /// implicitly mutates kernel-side state without requiring
    /// userspace `&mut`. `FsWatcher::poll_until`'s `&mut self`
    /// reborrows as `&self` here trivially.
    fn poll_once(
        &self,
        deadline: Option<Instant>,
        out: &mut Vec<WatcherEvent>,
    ) -> Result<usize, WatchFailure> {
        let mut events = [ffi::Kevent::zeroed(); EVENT_BATCH];
        // `kevent` may itself signal pressure (`EMFILE` from a full
        // kernel queue) in addition to the per-syscall errno set; route
        // every error through the typed boundary so the bin can demux on
        // the variant rather than re-classifying `io::Error` upstream.
        // Deadline tracking (including `EINTR`-retry remaining-budget
        // recompute) lives inside `kevent_drain`.
        let n = ffi::kevent_drain(&self.kq, &mut events, deadline)
            .map_err(|e| WatchFailure::from_io(&e))?;

        tracing::trace!(n, "kqueue drained");

        let mut emitted = 0usize;
        for ev in &events[..n] {
            // Wake events carry the EVFILT_USER ident and no ResourceId
            // payload — filter them silently before normalization.
            if ev.is_user_event(WAKE_IDENT) {
                continue;
            }
            let raw = ev.udata();
            if raw == 0 {
                tracing::trace!(?ev, "kevent with zero udata; dropped");
                continue;
            }
            let r = ResourceId::from(KeyData::from_ffi(raw));
            // Kind cache miss is possible if the resource was unwatched
            // between the kernel's queue-add and our drain; default to
            // `Unknown` and let `normalize` apply its defensive map.
            let kind = self
                .by_resource
                .get(r)
                .map_or(ResourceKind::Unknown, |e| e.kind);
            let Some(fs_event) = normalize::kevent_to_fs_event(ev.flags(), ev.fflags(), kind)
            else {
                continue;
            };
            // Stale-event passthrough: `r` may already have been removed
            // from `by_resource` between kernel queue-up and our drain.
            // Emit anyway — engine's `EventOnUnwatchedResource` Diagnostic
            // handles the race.
            out.push(WatcherEvent::Fs {
                resource: r,
                event: fs_event,
            });
            emitted += 1;
        }
        Ok(emitted)
    }
}

impl FsWatcher for KqueueWatcher {
    /// Trait wrapper around `Self::watch_inner`: classifies the inner
    /// `io::Error` into a typed [`WatchFailure`] at the boundary so the
    /// engine demuxes on the variant rather than on raw errno values.
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

    fn unwatch(&mut self, r: ResourceId) {
        // Drop the entry — `OwnedFd`'s `Drop` closes the fd, and the
        // kernel auto-removes the vnode registration as the fd closes.
        // The `(fd, fflags, kind)` triple lives in one struct, so a
        // single `remove(r)` covers all three pieces. Idempotent on
        // stale ids.
        let removed = self.by_resource.remove(r).is_some();
        tracing::debug!(?r, removed, "kqueue unwatch");
    }

    /// One blocking drain to the engine's deadline; see
    /// [`Self::poll_once`] for the per-call mechanics. The watcher does
    /// no event coalescing of its own — kqueue's `EV_CLEAR` already
    /// merges duplicate writes at the kernel level, and the engine's
    /// settle-timer reschedule debounces above the trait boundary.
    fn poll_until(
        &mut self,
        deadline: Option<Instant>,
        out: &mut Vec<WatcherEvent>,
    ) -> Result<usize, WatchFailure> {
        self.poll_once(deadline, out)
    }

    fn wake_handle(&self) -> Box<dyn WakeHandle> {
        Box::new(KqueueWakeHandle::new(Arc::clone(&self.kq), WAKE_IDENT))
    }
}
