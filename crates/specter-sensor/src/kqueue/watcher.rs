//! `KqueueWatcher` — kqueue-backed `FsWatcher` impl.
//!
//! Single-threaded: one thread owns the `KqueueWatcher` value and drives
//! `watch` / `unwatch` / `suppress` / `unsuppress` between `poll_until`
//! calls. The wake handle ([`KqueueWakeHandle`]) is the only cross-thread
//! surface — see [`crate::kqueue::wake`] for the lifecycle discipline.
//!
//! # Drop semantics
//!
//! Default field-order drop:
//! - `by_resource` drops every watched fd (kernel removes vnode
//!   registrations as each fd closes).
//! - `suppressed`, `kinds`, and `registered_fflags` drop their
//!   bookkeeping (no fds).
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
//! `unwatch` and `clamp_watch_demand_to_zero`-driven Unwatch ops.

use crate::kqueue::wake::KqueueWakeHandle;
use crate::kqueue::{fd, ffi, normalize, translate};
use crate::{FsWatcher, WakeHandle, WatchFailure, WatchFailureExt, WatcherEvent};
use slotmap::SecondaryMap;
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
    by_resource: SecondaryMap<ResourceId, OwnedFd>,
    suppressed: SecondaryMap<ResourceId, ()>,
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
            suppressed: SecondaryMap::new(),
            kinds: SecondaryMap::new(),
            registered_fflags: SecondaryMap::new(),
            kq,
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
        // iff different. The re-register composes EV_ADD | EV_CLEAR with
        // an optional EV_DISABLE on the **same change record**, so the
        // kernel-side filter never observes an enabled state mid-update
        // when re-registering on a previously suppressed FD. This closes
        // the two-syscall race that an EV_ADD-then-EV_DISABLE sequence
        // exposed (per kqueue(2) § EV_ADD: "Adding an event automatically
        // enables it, unless overridden by the EV_DISABLE flag").
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
            let suppressed = self.suppressed.contains_key(r);
            {
                let fd = self
                    .by_resource
                    .get(r)
                    .expect("by_resource.contains_key(r) was true");
                ffi::register_vnode(&self.kq, fd, r, new_fflags, !suppressed)?;
            }
            self.registered_fflags.insert(r, new_fflags);
            tracing::debug!(
                ?r,
                ?events,
                suppressed,
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
        // partially-failed `watch` leaves zero state. Fresh watches
        // start enabled — `suppressed` is populated by `suppress(r)`
        // after the WatchOp ordering puts a `Watch` before any
        // same-step `Suppress`, so any later silencing rides the
        // dedicated `disable_vnode` syscall path.
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
        ffi::register_vnode(&self.kq, &fd, r, fflags, true)?;
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
        self.suppressed.remove(r);
        self.kinds.remove(r);
        self.registered_fflags.remove(r);
        tracing::debug!(?r, removed, "kqueue unwatch");
    }

    fn suppress(&mut self, r: ResourceId) {
        let Some(fd) = self.by_resource.get(r) else {
            tracing::warn!(?r, "kqueue suppress on unwatched resource (race; dropped)");
            return;
        };
        // Pass the cached fflags so macOS's EV_DISABLE path preserves
        // the registered mask (FreeBSD ignores fflags on disable; macOS
        // overwrites). See `ffi::disable_vnode` for the platform note.
        let fflags = self.registered_fflags.get(r).copied().unwrap_or(0);
        if let Err(e) = ffi::disable_vnode(&self.kq, fd, r, fflags) {
            tracing::warn!(?r, error = ?e, "kqueue EV_DISABLE failed (likely race; dropped)");
            return;
        }
        self.suppressed.insert(r, ());
        tracing::debug!(?r, "kqueue suppress");
    }

    fn unsuppress(&mut self, r: ResourceId) {
        let Some(fd) = self.by_resource.get(r) else {
            tracing::warn!(
                ?r,
                "kqueue unsuppress on unwatched resource (race; dropped)"
            );
            return;
        };
        // See `suppress` — pass cached fflags so EV_ENABLE preserves
        // the registered mask on macOS.
        let fflags = self.registered_fflags.get(r).copied().unwrap_or(0);
        if let Err(e) = ffi::enable_vnode(&self.kq, fd, r, fflags) {
            tracing::warn!(?r, error = ?e, "kqueue EV_ENABLE failed (likely race; dropped)");
            return;
        }
        self.suppressed.remove(r);
        tracing::debug!(?r, "kqueue unsuppress");
    }

    fn poll_until(
        &mut self,
        deadline: Option<Instant>,
        out: &mut Vec<WatcherEvent>,
    ) -> Result<usize, WatchFailure> {
        let timeout = deadline.map(deadline_instant_to_timespec);
        let mut events = [ffi::Kevent::zeroed(); EVENT_BATCH];
        // `kevent` may itself signal pressure (`EMFILE` from a full
        // kernel queue) in addition to the per-syscall errno set; route
        // every error through the typed boundary so the bin can demux on
        // the variant rather than re-classifying `io::Error` upstream.
        let n = ffi::kevent_drain(&self.kq, &mut events, timeout)
            .map_err(|e| WatchFailure::from_io(&e))?;
        tracing::trace!(n, "kqueue drained");

        let mut emitted = 0usize;
        for ev in &events[..n] {
            // Wake events carry the EVFILT_USER ident and no ResourceId
            // payload — filter them silently before normalization.
            if ev.is_user_event(WAKE_IDENT) {
                continue;
            }
            let Some(r) = ev.resource_id() else {
                tracing::trace!(?ev, "kevent with unparseable udata; dropped");
                continue;
            };
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

/// Convert an `Instant` deadline to a kqueue-friendly `timespec`.
/// `d <= now` clamps to `ZERO` (non-blocking poll).
fn deadline_instant_to_timespec(d: Instant) -> libc::timespec {
    let dur = d.saturating_duration_since(Instant::now());
    ffi::duration_to_timespec(dur)
}

#[cfg(test)]
mod tests {
    use super::deadline_instant_to_timespec;
    use std::time::{Duration, Instant};

    #[test]
    fn deadline_in_past_clamps_to_zero() {
        let past = Instant::now()
            .checked_sub(Duration::from_mins(1))
            .expect("60s before Instant::now() is representable");
        let ts = deadline_instant_to_timespec(past);
        assert_eq!(ts.tv_sec, 0);
        assert_eq!(ts.tv_nsec, 0);
    }

    #[test]
    fn deadline_future_round_trip_within_a_second() {
        let dur = Duration::from_millis(500);
        let ts = deadline_instant_to_timespec(Instant::now() + dur);
        // The deadline is `now + 500ms` and `deadline_instant_to_timespec`
        // reads `Instant::now()` again internally, so the timespec is at
        // most 500ms and should be within ~50ms of that target.
        //
        // `tv_sec`/`tv_nsec` are signed (`i64`/`c_long`) on macOS/FreeBSD;
        // the conversion back to `u64`/`u32` is bounded by the sub-second
        // duration we just produced.
        let secs = u64::try_from(ts.tv_sec).expect("non-negative tv_sec");
        let nanos = u32::try_from(ts.tv_nsec).expect("non-negative, < 1s tv_nsec");
        let dur_ts = Duration::new(secs, nanos);
        assert!(dur_ts <= dur, "{dur_ts:?} <= {dur:?}");
        assert!(
            dur_ts > dur.saturating_sub(Duration::from_millis(50)),
            "{dur_ts:?} > {:?}",
            dur.saturating_sub(Duration::from_millis(50))
        );
    }
}
