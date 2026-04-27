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
//! - `suppressed` and `kinds` drop their bookkeeping (no fds).
//! - `kq` (Arc) decrements; if last, the kqueue fd closes, kernel-reaping
//!   the `EVFILT_USER` ident and any queued events.
//!
//! Wake handles holding Arc clones keep the kqueue fd alive past the
//! watcher's drop — `wake()` from those becomes a no-op-equivalent (no
//! consumer drains the resulting event), with no UB.

use crate::kqueue::wake::KqueueWakeHandle;
use crate::kqueue::{fd, ffi, normalize};
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
    /// `NOTE_WRITE` on Dir vs File. Mirrors the engine's `Resource.kind`
    /// independently — drift between the two is acceptable (the engine
    /// uses its `kind` for `covers` / `EffectScope` semantics; the
    /// watcher uses this one purely for event normalization).
    kinds: SecondaryMap<ResourceId, ResourceKind>,
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
            kq,
        })
    }
}

impl FsWatcher for KqueueWatcher {
    fn watch(&mut self, r: ResourceId, path: &Path, _opts: WatchOpts) -> io::Result<()> {
        // 1) Open. 2) Stat. 3) Register. 4) Insert. Each step's failure
        //    drops anything earlier (the OwnedFd auto-closes), so a
        //    partially-failed `watch` leaves zero state.
        let fd = fd::open_for_watch(path)?;
        let kind = fd::stat_kind(&fd)?;
        ffi::register_vnode(&self.kq, &fd, r)?;
        self.by_resource.insert(r, fd);
        self.kinds.insert(r, kind);
        // Fresh watch starts unsuppressed by construction; nothing to
        // do on `suppressed`.
        tracing::debug!(?r, ?path, ?kind, "kqueue watch");
        Ok(())
    }

    fn unwatch(&mut self, r: ResourceId) {
        // Drop the OwnedFd — kernel auto-removes the vnode registration
        // when the fd closes. Idempotent on stale ids.
        let removed = self.by_resource.remove(r).is_some();
        self.suppressed.remove(r);
        self.kinds.remove(r);
        tracing::debug!(?r, removed, "kqueue unwatch");
    }

    fn suppress(&mut self, r: ResourceId) {
        let Some(fd) = self.by_resource.get(r) else {
            tracing::warn!(?r, "kqueue suppress on unwatched resource (race; dropped)");
            return;
        };
        if let Err(e) = ffi::disable_vnode(&self.kq, fd, r) {
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
        if let Err(e) = ffi::enable_vnode(&self.kq, fd, r) {
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
