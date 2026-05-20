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
//! - `by_resource` drops every watched fd (kernel removes vnode
//!   registrations as each fd closes).
//! - `kinds` and `registered_fflags` drop their bookkeeping (no fds).
//! - `kq` (Arc) decrements; if last, the kqueue fd closes, kernel-reaping
//!   the `EVFILT_USER` ident and any queued events.
//!
//! Wake handles holding Arc clones keep the kqueue fd alive past the
//! watcher's drop — `wake()` from those becomes a no-op-equivalent (no
//! consumer drains the resulting event), with no UB.
//!
//! # Per-FD mask cache
//!
//! The engine emits `WatchOp::Watch` whenever `Resource.events_union`
//! changes, not only on the 0→1 refcount edge.
//! The watcher caches the post-translation kqueue fflags per resource
//! (`registered_fflags`) so a re-`watch()` with an unchanged mask skips
//! the syscall entirely, and a re-`watch()` with a widened/narrowed mask
//! re-registers via `EV_ADD` (which overwrites the prior fflags) without
//! closing or reopening the fd. The cache is invalidated only on
//! `unwatch` (the engine-side `Unwatch` op, sourced from `sub_watch`'s
//! non-empty → empty edge or `Tree::vacate`'s terminus emission).

use crate::kqueue::wake::KqueueWakeHandle;
use crate::kqueue::{fd, ffi, normalize, translate};
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
    by_resource: SecondaryMap<ResourceId, OwnedFd>,
    /// Per-resource kind cache: seeded at `watch()` from the engine's
    /// `WatchOp::Watch.kind` (verified against an `fstat` of the freshly
    /// opened fd; the cache stores the verified value). Consumed by
    /// `normalize::kevent_to_fs_event` to disambiguate `NOTE_WRITE` on
    /// Dir vs File and by `translate::class_set_to_fflags` to compute
    /// the per-FD mask. The verification step closes the TOCTOU window
    /// between the engine's classification and the kernel's
    /// path-resolution at watch-install time.
    kinds: SecondaryMap<ResourceId, ResourceKind>,
    /// Per-resource kqueue fflags cache: populated alongside `by_resource`
    /// from the translator's output (`class_set_to_fflags(events,
    /// kind)`). Used by `watch()` to diff the incoming mask against the
    /// installed one so unchanged re-registrations skip the syscall, and
    /// changed ones re-register via `EV_ADD` without reopening the fd.
    /// Cleared in lockstep with `by_resource` on `unwatch()`.
    registered_fflags: SecondaryMap<ResourceId, u32>,
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
            kinds: SecondaryMap::new(),
            registered_fflags: SecondaryMap::new(),
            kq,
            drain_window,
            last_event_at: None,
        })
    }

    /// Internal `watch` body returning the raw `io::Error` set; the
    /// trait wrapper maps that into a typed [`WatchFailure`] at the
    /// boundary so internal `?` propagation stays uniform.
    ///
    /// Two paths share this entry point: a fresh-watch (no FD held for
    /// `r`) and a re-watch (engine emitted a fresh `WatchOp::Watch`
    /// because `Resource.events_union` changed at non-zero refcount).
    /// The two diverge on whether `by_resource` already holds an
    /// `OwnedFd` for `r`; the re-watch path skips open/stat and reuses
    /// the existing FD, diffing the cached fflags against the
    /// translator's output.
    fn watch_inner(
        &mut self,
        r: ResourceId,
        path: &Path,
        kind: ResourceKind,
        events: ClassSet,
    ) -> io::Result<()> {
        // ── Re-watch path ───────────────────────────────────────────
        // FD already held: compute the new fflags from the cached kind
        // + incoming events; diff against the installed mask; re-register
        // iff different.
        //
        // The engine-supplied `kind` is ignored on the re-watch path:
        // the cached kind (verified against `fstat` at fresh-watch time)
        // is the authoritative value and is invariant for the FD's
        // lifetime — re-watch never reopens.
        if self.by_resource.contains_key(r) {
            let cached_kind = self.kinds.get(r).copied().unwrap_or(ResourceKind::Unknown);
            let new_fflags = translate::class_set_to_fflags(events, cached_kind);
            let cached_fflags = self.registered_fflags.get(r).copied().unwrap_or(0);
            if new_fflags == cached_fflags {
                tracing::trace!(
                    ?r,
                    ?events,
                    fflags = format_args!("{cached_fflags:#x}"),
                    "kqueue re-watch noop (mask unchanged)"
                );
                return Ok(());
            }
            {
                let fd = self
                    .by_resource
                    .get(r)
                    .expect("by_resource.contains_key(r) was true");
                ffi::register_vnode(&self.kq, fd, r.data().as_ffi(), new_fflags)?;
            }
            self.registered_fflags.insert(r, new_fflags);
            tracing::debug!(
                ?r,
                ?events,
                old_fflags = format_args!("{cached_fflags:#x}"),
                new_fflags = format_args!("{new_fflags:#x}"),
                "kqueue re-register (mask changed)"
            );
            return Ok(());
        }

        // ── Fresh-watch path ────────────────────────────────────────
        // 1) Open. 2) Stat + verify against engine's expected kind.
        // 3) Translate. 4) Register. 5) Insert. Each step's failure
        // drops anything earlier (the OwnedFd auto-closes) so a
        // partially-failed `watch` leaves zero state.
        let fd = fd::open_for_watch(path)?;
        let observed_kind = fd::stat_kind(&fd)?;
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
        let fflags = translate::class_set_to_fflags(events, observed_kind);
        ffi::register_vnode(&self.kq, &fd, r.data().as_ffi(), fflags)?;
        self.by_resource.insert(r, fd);
        self.kinds.insert(r, observed_kind);
        self.registered_fflags.insert(r, fflags);
        tracing::debug!(
            ?r,
            ?path,
            kind = ?observed_kind,
            ?events,
            fflags = format_args!("{fflags:#x}"),
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
        // Drop the OwnedFd — kernel auto-removes the vnode registration
        // when the fd closes. Idempotent on stale ids. The fflags cache
        // tracks the FD's lifetime exactly: clear it whenever we drop
        // the FD so a subsequent re-watch starts fresh.
        let removed = self.by_resource.remove(r).is_some();
        self.kinds.remove(r);
        self.registered_fflags.remove(r);
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
    /// **Recency gate (`last_event_at`).** Phase 2 enters iff:
    /// 1. Phase 1 returned at least one **real** (non-wake) event,
    /// 2. The drain window is non-zero, AND
    /// 3. The prior drain that returned real events was within one
    ///    drain window of `now` (`now - last_event_at < window`).
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

        // Filter wake events for the recency check. A wake-only return
        // means the bin pushed fresh `WatchOp`s through the channel —
        // file traffic hasn't materialised, so the burst-cadence
        // heuristic mustn't update its timestamp on this drain.
        let phase1_real = events[..n1]
            .iter()
            .filter(|ev| !ev.is_user_event(WAKE_IDENT))
            .count();

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

            // Phase 2 enters only if we have buffer space, the window
            // is enabled, and the prior drain was within window. The
            // buffer-full case still updates `last_event_at` above so
            // the next iteration's recency gate opens — pent-up
            // events drain on the follow-up call.
            if n1 < EVENT_BATCH && recent && window > Duration::ZERO {
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
            let kind = self.kinds.get(r).copied().unwrap_or(ResourceKind::Unknown);
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
