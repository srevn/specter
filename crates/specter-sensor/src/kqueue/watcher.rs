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
//! Under R2 / D11, the engine emits `WatchOp::Watch` whenever
//! `Resource.events_union` changes, *not* only on the 0→1 refcount edge.
//! The watcher caches the post-translation kqueue fflags per resource
//! (`registered_fflags`) so a re-`watch()` with an unchanged mask skips
//! the syscall entirely, and a re-`watch()` with a widened/narrowed mask
//! re-registers via `EV_ADD` (which overwrites the prior fflags) without
//! closing or reopening the fd. The cache is invalidated only on
//! `unwatch` and `clamp_watch_demand_to_zero`-driven Unwatch ops.

use crate::kqueue::wake::KqueueWakeHandle;
use crate::kqueue::{fd, ffi, normalize, translate};
use crate::{FsWatcher, WakeHandle};
use slotmap::SecondaryMap;
use specter_core::{FsEvent, ResourceId, ResourceKind, WatchOpts};
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
    /// Per-resource kind cache: populated at `watch()` from `fstat`,
    /// consumed by `normalize::kevent_to_fs_event` to disambiguate
    /// `NOTE_WRITE` on Dir vs File and by `translate::class_set_to_fflags`
    /// to compute the per-FD mask. Mirrors the engine's `Resource.kind`
    /// independently — drift between the two is acceptable (the engine
    /// uses its `kind` for `covers` / `EffectScope` semantics; the
    /// watcher uses this one purely for event normalization and mask
    /// translation).
    kinds: SecondaryMap<ResourceId, ResourceKind>,
    /// Per-resource kqueue fflags cache: populated alongside `by_resource`
    /// from the L4 translator's output (`class_set_to_fflags(opts.events,
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
}

impl FsWatcher for KqueueWatcher {
    /// Two paths share this entry point: a fresh-watch (no FD held for
    /// `r`) and a re-watch (engine emitted a fresh `WatchOp::Watch`
    /// because `Resource.events_union` changed at non-zero refcount, per
    /// D11). The two diverge on whether `by_resource` already holds an
    /// `OwnedFd` for `r`; the re-watch path skips open/stat and reuses
    /// the existing FD, diffing the cached fflags against the
    /// translator's output.
    fn watch(&mut self, r: ResourceId, path: &Path, opts: WatchOpts) -> io::Result<()> {
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
        if self.by_resource.contains_key(r) {
            let kind = self.kinds.get(r).copied().unwrap_or(ResourceKind::Unknown);
            let new_fflags = translate::class_set_to_fflags(opts.events, kind);
            let cached_fflags = self.registered_fflags.get(r).copied().unwrap_or(0);
            if new_fflags == cached_fflags {
                tracing::trace!(
                    ?r,
                    ?opts,
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
                ?opts,
                suppressed,
                old_fflags = format_args!("{cached_fflags:#x}"),
                new_fflags = format_args!("{new_fflags:#x}"),
                "kqueue re-register (mask changed)"
            );
            return Ok(());
        }

        // ── Fresh-watch path ────────────────────────────────────────
        // 1) Open. 2) Stat. 3) Translate. 4) Register. 5) Insert. Each
        // step's failure drops anything earlier (the OwnedFd auto-closes)
        // so a partially-failed `watch` leaves zero state. Fresh watches
        // start enabled — `suppressed` is populated by `suppress(r)`
        // after the WatchOp ordering puts a `Watch` before any same-step
        // `Suppress`, so any later silencing rides the dedicated
        // `disable_vnode` syscall path.
        let fd = fd::open_for_watch(path)?;
        let kind = fd::stat_kind(&fd)?;
        let fflags = translate::class_set_to_fflags(opts.events, kind);
        ffi::register_vnode(&self.kq, &fd, r, fflags, true)?;
        self.by_resource.insert(r, fd);
        self.kinds.insert(r, kind);
        self.registered_fflags.insert(r, fflags);
        tracing::debug!(
            ?r,
            ?path,
            ?kind,
            ?opts,
            fflags = format_args!("{fflags:#x}"),
            "kqueue watch"
        );
        Ok(())
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
        out: &mut Vec<(ResourceId, FsEvent)>,
    ) -> io::Result<usize> {
        let timeout = deadline.map(deadline_instant_to_timespec);
        let mut events = [ffi::Kevent::zeroed(); EVENT_BATCH];
        let n = ffi::kevent_drain(&self.kq, &mut events, timeout)?;
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
            out.push((r, fs_event));
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
pub(super) fn deadline_instant_to_timespec(d: Instant) -> libc::timespec {
    let dur = d.saturating_duration_since(Instant::now());
    ffi::duration_to_timespec(dur)
}
