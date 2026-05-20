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
use crate::{DrainWindow, FsWatcher, WakeHandle, WatchFailure, WatchFailureExt, WatcherEvent};
use slotmap::{Key, KeyData, SecondaryMap};
use specter_core::{ClassSet, ResourceId, ResourceKind};
use std::io;
use std::os::fd::OwnedFd;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

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
    /// Cross-thread, fixed drain window. The bin constructs it once and
    /// hands it over; this watcher reads it on every `poll_until`
    /// iteration via [`DrainWindow::get`]. Never mutated at runtime.
    /// `Duration::ZERO` disables the deferred drain entirely.
    drain_window: DrainWindow,
    /// Timestamp of the most recent drain that returned at least one
    /// real (non-wake) event. The recency gate for the deferred-drain
    /// phase reads this; a fresh-burst drain (no prior timestamp) or a
    /// long-quiet drain (`now - last_event_at >= drain_window`) leaves
    /// the gate closed and the second drain pass is skipped.
    ///
    /// `None` until the first non-wake-only `poll_until` return; held
    /// across the watcher's lifetime so a quiet-then-burst pattern
    /// re-engages on the second drain of the burst (single touches in
    /// the gap stay fast).
    last_event_at: Option<Instant>,
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
    ///
    /// `drain_window` shapes the deferred-drain pass in `poll_until`;
    /// see [`DrainWindow`] for the semantics. The handle is stored as
    /// an `Arc` clone — cheap per construction.
    pub fn new(drain_window: DrainWindow) -> io::Result<Self> {
        let kq = Arc::new(ffi::kqueue_new()?);
        ffi::register_user_event(&kq, WAKE_IDENT)?;
        Ok(Self {
            by_resource: SecondaryMap::new(),
            kq,
            drain_window,
            last_event_at: None,
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

    /// Block until events arrive (or the deadline elapses or a wake
    /// fires), then optionally arm a second `kevent_drain` to capture
    /// any kernel-queued events arriving within the drain window.
    ///
    /// **Two drain phases.** Phase 1 is the engine-driven blocking
    /// drain that returns on the first kernel signal. Phase 2 is the
    /// *deferred* drain — a short follow-up `kevent` that lets a
    /// kernel-coalesced event burst surface in one `poll_until`
    /// iteration instead of fragmenting across many.
    ///
    /// **Phase-2 gate.** Phase 2 enters iff every term holds:
    /// 1. Phase 1 returned at least one **real** (non-wake) event,
    /// 2. Phase 1 had buffer space remaining (`n1 < EVENT_BATCH`),
    /// 3. The drain window is non-zero,
    /// 4. The prior drain that returned real events was within one
    ///    drain window of `now` (`now - last_event_at < window`),
    /// 5. **No wake fired in phase 1.** A wake observed alongside
    ///    real events signals the bin pushed fresh `WatchOp`s through
    ///    the channel; deferring the return to drain again would
    ///    delay the bin's loop iteration. The watcher returns
    ///    promptly so pending control-plane work is applied before
    ///    the next blocking drain.
    ///
    /// Together these gates keep the latency cost out of the
    /// single-event-quiet-period path: the first event of a fresh
    /// burst (or the only event of a quiet workload like W_edit) sees
    /// `last_event_at` stale or unset and skips phase 2 entirely.
    /// Sustained bursts (W_ssh / W_build) catch phase 2 from the
    /// second drain in the burst onwards, batching the kernel's
    /// coalesce stream into the engine's debounce window.
    ///
    /// **Buffer-full short-circuit.** When phase 1 fills `EVENT_BATCH`,
    /// phase 2 is skipped (no buffer space). The kernel queue retains
    /// the excess; the next `poll_until` iteration drains them, with
    /// `last_event_at` updated so the recency gate opens.
    ///
    /// **Hot reload.** [`DrainWindow::get`] is read once per drain
    /// iteration; subsequent updates apply to the next call. At most
    /// one drain straddles a reload.
    fn poll_until(
        &mut self,
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
        let n1 = ffi::kevent_drain(&self.kq, &mut events, deadline)
            .map_err(|e| WatchFailure::from_io(&e))?;

        // Single pass over the drained batch: count real (non-wake)
        // events and detect whether a wake fired. A wake-only return
        // means the bin pushed fresh `WatchOp`s through the channel —
        // file traffic hasn't materialised, so the burst-cadence
        // heuristic mustn't update its timestamp on this drain. A
        // wake alongside real events signals the same pending
        // control-plane work; phase 2 is suppressed so the return
        // reaches the bin promptly.
        let (phase1_real, phase1_woke) =
            events[..n1]
                .iter()
                .fold((0usize, false), |(real, woke), ev| {
                    if ev.is_user_event(WAKE_IDENT) {
                        (real, true)
                    } else {
                        (real + 1, woke)
                    }
                });

        let n_total = if phase1_real > 0 {
            let now = Instant::now();
            let window = self.drain_window.get();
            // Recency check against the *prior* drain's timestamp, then
            // update — so the first drain of a fresh burst always
            // skips phase 2 (no prior timestamp ⇒ `recent == false`).
            let recent = self
                .last_event_at
                .is_some_and(|t| now.saturating_duration_since(t) < window);
            self.last_event_at = Some(now);

            // Phase-2 gate: buffer space, window enabled, prior drain
            // within window, AND no concurrent wake. The buffer-full
            // case still updates `last_event_at` above so the next
            // iteration's recency gate opens — pent-up events drain
            // on the follow-up call.
            if n1 < EVENT_BATCH && recent && window > Duration::ZERO && !phase1_woke {
                // Cap the phase-2 deadline at `now + window`; an
                // engine-supplied deadline already tighter than that
                // wins (timer cadence is preserved even on a
                // window-deferred drain). `kevent_drain` recomputes
                // the remaining budget on every `EINTR` retry.
                let phase2_deadline = deadline.map_or(now + window, |d| d.min(now + window));
                let n2 = ffi::kevent_drain(&self.kq, &mut events[n1..], Some(phase2_deadline))
                    .map_err(|e| WatchFailure::from_io(&e))?;
                n1 + n2
            } else {
                n1
            }
        } else {
            n1
        };

        tracing::trace!(n1, n_total, "kqueue drained");

        let mut emitted = 0usize;
        for ev in &events[..n_total] {
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
            //
            // kqueue never emits `WatcherEvent::Overflow` under v1: the
            // EV_CLEAR coalesce semantic merges duplicate writes into one
            // delivered event, but it never silently drops. Overflow is
            // an inotify-only concept (`IN_Q_OVERFLOW`); the bin's loop
            // is shaped to accept either variant from any backend.
            out.push(WatcherEvent::Fs {
                resource: r,
                event: fs_event,
            });
            emitted += 1;
        }
        Ok(emitted)
    }

    fn wake_handle(&self) -> Box<dyn WakeHandle> {
        Box::new(KqueueWakeHandle::new(Arc::clone(&self.kq), WAKE_IDENT))
    }
}
