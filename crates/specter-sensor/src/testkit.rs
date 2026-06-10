//! Pure-Rust `FsWatcher` test fixture.
//!
//! Records every mutator call into a `Vec<WatcherCall>`; tests inject events for `drain_ready` to
//! deliver synchronously. The mock owns a `UnixStream::pair()` readiness substrate: every `inject`
//! / `inject_overflow` writes one byte to the write side, making the read side (the `AsFd` surface)
//! readable; `drain_ready` reads the substrate to empty before draining the queued events. This
//! pins the edge-triggered readiness contract `FsWatcher` requires — a reactor (mio::Poll,
//! libc::poll, etc.) registering [`AsFd::as_fd`] observes the same drain-or-stall semantics as a
//! real kqueue / inotify fd.
//!
//! No FFI, no kqueue, no platform gates — compiles on every Unix target (Linux CI, macOS dev,
//! FreeBSD prod). Production builds opt out via the `testkit` Cargo feature; consumers attach it
//! under `[dev-dependencies]`.

use crate::{FsWatcher, OverflowScope, Prober, WatchFailure, WatcherEvent};
use slotmap::SecondaryMap;
use specter_core::{ClassSet, FsEvent, ProbeOwner, ProbeRequest, ResourceId, ResourceKind};
use std::io::{self, Read, Write};
use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

/// One recorded call into the watcher's mutator surface.
#[derive(Debug, Clone)]
pub enum WatcherCall {
    Watch {
        resource: ResourceId,
        path: PathBuf,
        kind: ResourceKind,
        events: ClassSet,
    },
    Unwatch {
        resource: ResourceId,
    },
}

/// Live record of an installed watch on the mock. One entry per successfully-`watch()`ed resource;
/// cleared on `unwatch()`. Tests inspect this for the post-install state without walking the call
/// log.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MockEntry {
    pub path: PathBuf,
    pub kind: ResourceKind,
    pub events: ClassSet,
}

#[derive(Debug)]
pub struct MockFsWatcher {
    /// Append-only log of every mutator call. Ordering is the order of invocation — useful for
    /// asserting cross-step sequences.
    pub calls: Vec<WatcherCall>,
    /// Current installed-watch state: `r ∈ installed` ⇔ last `watch(r)` succeeded and no subsequent
    /// `unwatch(r)` cleared it.
    pub installed: SecondaryMap<ResourceId, MockEntry>,
    /// Per-resource events queued for delivery on the next `drain_ready` call. Drained into
    /// [`WatcherEvent::Fs`] before any queued overflow scopes (preserves the natural "events first,
    /// kernel signals last" ordering the bin's loop expects).
    pub queued_events: Vec<(ResourceId, FsEvent)>,
    /// Overflow scopes queued for delivery on the next `drain_ready` call. Drained into
    /// [`WatcherEvent::Overflow`] **after** the per-resource events. Tests use this to exercise the
    /// bin's overflow-routing path without wiring a real inotify backend.
    pub queued_overflow: Vec<OverflowScope>,
    /// Set by `fail_next_watch` to simulate FD pressure / kind mismatch / programmer error — the
    /// **next** `watch()` call returns `Err(failure)` without modifying state. One-shot; consumed
    /// on read.
    pub next_watch_failure: Option<WatchFailure>,
    /// Read side of the readiness substrate, set non-blocking. A reactor (mio::Poll, libc::poll)
    /// registering [`AsFd::as_fd`] sees this fd; it becomes readable when `inject` /
    /// `inject_overflow` has written to `write_fd` and `drain_ready` has not yet drained it.
    /// Edge-triggered semantics require `drain_ready` to read to empty before returning — otherwise
    /// the next inject would not trigger a fresh edge.
    read_fd: UnixStream,
    /// Write side of the readiness substrate. Bumped one byte per `inject` / `inject_overflow` to
    /// drive a readability edge on `read_fd`. Set non-blocking so a runaway test doesn't deadlock
    /// the writer on a full socket buffer; at test scale (single-digit bytes between drains, socket
    /// buffer ≥ 4 KiB) the path is unreachable.
    write_fd: UnixStream,
}

impl MockFsWatcher {
    /// Construct a fresh mock. Allocates a `socketpair(AF_UNIX, SOCK_STREAM)` for the readiness
    /// substrate; panics on any syscall failure — testkit code is test-only, and an unrecoverable
    /// kernel error here is properly fatal.
    #[must_use]
    pub fn new() -> Self {
        let (read_fd, write_fd) =
            UnixStream::pair().expect("MockFsWatcher: socketpair must succeed in tests");
        read_fd
            .set_nonblocking(true)
            .expect("MockFsWatcher: O_NONBLOCK on read side");
        write_fd
            .set_nonblocking(true)
            .expect("MockFsWatcher: O_NONBLOCK on write side");
        Self {
            calls: Vec::new(),
            installed: SecondaryMap::new(),
            queued_events: Vec::new(),
            queued_overflow: Vec::new(),
            next_watch_failure: None,
            read_fd,
            write_fd,
        }
    }

    /// Queue a per-resource event for delivery on the next `drain_ready` call. The queue drains
    /// entirely on drain — caller is responsible for re-injecting between successive drains. Bumps
    /// the readiness substrate so a reactor on [`AsFd::as_fd`] sees the fd transition to readable.
    pub fn inject(&mut self, r: ResourceId, ev: FsEvent) {
        self.queued_events.push((r, ev));
        self.bump_fd();
    }

    /// Queue an overflow signal for delivery on the next `drain_ready` call. Drained into
    /// [`WatcherEvent::Overflow`] **after** any queued per-resource events on the same drain,
    /// mirroring the inotify backend's natural ordering (`Fs` events for the records preceding the
    /// overflow marker; `Overflow` once the kernel signals the queue ran dry). Bumps the readiness
    /// substrate so a reactor on [`AsFd::as_fd`] sees the fd transition to readable.
    ///
    /// Useful for engine / bin tests that need to assert the `Overflow` routing path without
    /// spinning up a real inotify instance and stressing it past `max_queued_events`.
    pub fn inject_overflow(&mut self, scope: OverflowScope) {
        self.queued_overflow.push(scope);
        self.bump_fd();
    }

    /// Cause the next `watch` call to fail with `failure`. Consumed on read — subsequent watches
    /// succeed unless re-armed.
    pub const fn fail_next_watch(&mut self, failure: WatchFailure) {
        self.next_watch_failure = Some(failure);
    }

    /// Currently-registered event-class mask for `r`, derived from the last lifecycle op in `calls`:
    ///
    /// - Returns `Some(events)` if the most recent op for `r` is a `Watch` (regardless of whether
    ///   that `Watch` succeeded — see [`fail_next_watch`](Self::fail_next_watch)).
    /// - Returns `None` if the most recent op is `Unwatch` or if `r` has no lifecycle ops in the
    ///   call log.
    ///
    /// The engine emits a fresh `WatchOp::Watch` whenever `Resource.events_union` widens or
    /// narrows, so the latest `Watch` reflects the engine's current per-Resource union.
    #[must_use]
    pub fn registered_events(&self, r: ResourceId) -> Option<ClassSet> {
        for c in self.calls.iter().rev() {
            match c {
                WatcherCall::Watch {
                    resource, events, ..
                } if *resource == r => {
                    return Some(*events);
                }
                WatcherCall::Unwatch { resource } if *resource == r => return None,
                _ => {}
            }
        }
        None
    }

    /// Write one byte to `write_fd` to drive a readability edge on `read_fd`. Best-effort: a
    /// saturated socket buffer would surface as `WouldBlock`, but at test scale (single-digit bytes
    /// between drains, socket buffer ≥ 4 KiB) the path is unreachable. Failure is silently ignored
    /// — the test would then never see a readiness edge and watchdog out, which is the right
    /// failure mode.
    fn bump_fd(&self) {
        let _ = (&self.write_fd).write(&[0x01]);
    }
}

impl Default for MockFsWatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl FsWatcher for MockFsWatcher {
    fn watch(
        &mut self,
        r: ResourceId,
        path: &Path,
        kind: ResourceKind,
        events: ClassSet,
    ) -> Result<(), WatchFailure> {
        self.calls.push(WatcherCall::Watch {
            resource: r,
            path: path.to_owned(),
            kind,
            events,
        });
        if let Some(failure) = self.next_watch_failure.take() {
            return Err(failure);
        }
        self.installed.insert(
            r,
            MockEntry {
                path: path.to_owned(),
                kind,
                events,
            },
        );
        Ok(())
    }

    fn unwatch(&mut self, r: ResourceId) {
        self.calls.push(WatcherCall::Unwatch { resource: r });
        self.installed.remove(r);
    }

    fn drain_ready(&mut self, out: &mut Vec<WatcherEvent>) -> Result<usize, WatchFailure> {
        // 1. Drain the readiness substrate to empty. Edge-triggered reactors require this: leaving
        //    residual bytes strands the next event behind a missing not-readable→readable edge.
        let mut sink = [0u8; 64];
        loop {
            match (&self.read_fd).read(&mut sink) {
                // Got bytes — re-loop to drain more.
                Ok(n) if n > 0 => {}
                // `Ok(0)` is socket EOF — unreachable here because we own `write_fd` and it lives
                // as long as `self`. Treat as drained anyway to keep the loop terminal under any
                // future refactor.
                Ok(_) => break,
                // EAGAIN under O_NONBLOCK: substrate is empty.
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                // Spurious EINTR — re-loop. Highly improbable on a non-blocking socket (the kernel
                // returns before any signal can preempt), but cheap defence-in-depth.
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                // Mock — swallow any other error rather than fail the drain; the queued-events path
                // below still delivers.
                Err(_) => break,
            }
        }
        // 2. Drain queued events.
        let fs = std::mem::take(&mut self.queued_events);
        let overflow = std::mem::take(&mut self.queued_overflow);
        let emitted = fs.len() + overflow.len();
        out.extend(
            fs.into_iter()
                .map(|(resource, event)| WatcherEvent::Fs { resource, event }),
        );
        out.extend(
            overflow
                .into_iter()
                .map(|scope| WatcherEvent::Overflow { scope }),
        );
        Ok(emitted)
    }
}

impl AsFd for MockFsWatcher {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.read_fd.as_fd()
    }
}

/// Pure-Rust [`Prober`] test fixture.
///
/// Records every `submit` and `cancel` call into `Mutex<Vec<...>>`s for assertion. Does *not*
/// synthesize `ProbeResponse`s — engine tests inject responses via
/// `core::testkit::sensor::MockSensor`'s `probe_response` constructor directly.
///
/// `MockProber` is the right fixture for *bin / integration* tests that need to assert "the engine
/// emitted these probe operations to the prober" without spinning up a real `WorkerProber` against
/// a `tempfile::TempDir`. Real-fs round-trip tests use `WorkerProber`.
#[derive(Debug, Default)]
pub struct MockProber {
    pub submitted: Mutex<Vec<ProbeRequest>>,
    pub cancelled: Mutex<Vec<ProbeOwner>>,
}

impl MockProber {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drain the recorded `submit` calls.
    #[must_use]
    pub fn take_submitted(&self) -> Vec<ProbeRequest> {
        std::mem::take(&mut self.submitted.lock().expect("MockProber poisoned"))
    }

    /// Drain the recorded `cancel` calls.
    #[must_use]
    pub fn take_cancelled(&self) -> Vec<ProbeOwner> {
        std::mem::take(&mut self.cancelled.lock().expect("MockProber poisoned"))
    }
}

impl Prober for MockProber {
    fn submit(&self, req: ProbeRequest) {
        self.submitted
            .lock()
            .expect("MockProber poisoned")
            .push(req);
    }

    fn cancel(&self, owner: ProbeOwner) {
        self.cancelled
            .lock()
            .expect("MockProber poisoned")
            .push(owner);
    }
}

#[cfg(test)]
mod tests {
    use super::{MockFsWatcher, MockProber, WatcherCall};
    use crate::{FsWatcher, OverflowScope, Prober, WatchFailure, WatcherEvent};
    use slotmap::SlotMap;
    use specter_core::{
        ClassSet, FsEvent, ProbeCorrelation, ProbeOwner, ProbeRequest, ProfileId, ResourceId,
        ResourceKind,
    };
    use std::os::fd::AsFd;
    use std::path::PathBuf;
    use std::sync::Arc;

    fn fresh_resource_ids(n: usize) -> Vec<ResourceId> {
        let mut sm = SlotMap::<ResourceId, ()>::with_key();
        (0..n).map(|_| sm.insert(())).collect()
    }

    #[test]
    fn watch_records_call_and_installs() {
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::Unknown,
            ClassSet::EMPTY,
        )
        .expect("watch ok");

        assert_eq!(w.calls.len(), 1);
        assert!(matches!(
            &w.calls[0],
            WatcherCall::Watch { resource, .. } if *resource == ids[0]
        ));
        let entry = w.installed.get(ids[0]).expect("installed");
        assert_eq!(entry.path, PathBuf::from("/tmp/a"));
        assert_eq!(entry.kind, ResourceKind::Unknown);
        assert_eq!(entry.events, ClassSet::EMPTY);
    }

    #[test]
    fn watch_returns_error_when_armed() {
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();
        w.fail_next_watch(WatchFailure::Pressure {
            errno: libc::EMFILE,
        });

        let res = w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::Unknown,
            ClassSet::EMPTY,
        );
        assert_eq!(
            res,
            Err(WatchFailure::Pressure {
                errno: libc::EMFILE
            })
        );
        // The call is still recorded; the error short-circuits installation.
        assert_eq!(w.calls.len(), 1);
        assert!(!w.installed.contains_key(ids[0]));

        // One-shot — the next watch succeeds.
        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::Unknown,
            ClassSet::EMPTY,
        )
        .expect("second watch ok");
        assert!(w.installed.contains_key(ids[0]));
    }

    #[test]
    fn unwatch_clears_installed() {
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::Unknown,
            ClassSet::EMPTY,
        )
        .unwrap();
        assert!(w.installed.contains_key(ids[0]));

        w.unwatch(ids[0]);
        assert!(!w.installed.contains_key(ids[0]));
    }

    #[test]
    fn drain_ready_drains_queued_events_as_fs_variant() {
        let ids = fresh_resource_ids(2);
        let mut w = MockFsWatcher::new();

        w.inject(ids[0], FsEvent::ContentChanged);
        w.inject(ids[1], FsEvent::Renamed);

        let mut out: Vec<WatcherEvent> = Vec::new();
        let n = w.drain_ready(&mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out.len(), 2);
        assert!(w.queued_events.is_empty());
        assert!(matches!(
            &out[0],
            WatcherEvent::Fs { resource, event } if *resource == ids[0] && *event == FsEvent::ContentChanged
        ));
        assert!(matches!(
            &out[1],
            WatcherEvent::Fs { resource, event } if *resource == ids[1] && *event == FsEvent::Renamed
        ));

        // Second drain: nothing queued.
        out.clear();
        let n = w.drain_ready(&mut out).unwrap();
        assert_eq!(n, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn drain_ready_drains_queued_overflow_after_fs_events() {
        // Ordering invariant: any per-resource events queued before the drain surface as `Fs`
        // records first, then the queued overflow scopes as `Overflow` records. This mirrors the
        // inotify backend's natural drain order — events for records preceding the overflow marker
        // come first, then `IN_Q_OVERFLOW`.
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.inject(ids[0], FsEvent::ContentChanged);
        w.inject_overflow(OverflowScope::Global);
        w.inject_overflow(OverflowScope::Resource(ids[0]));

        let mut out: Vec<WatcherEvent> = Vec::new();
        let n = w.drain_ready(&mut out).unwrap();
        assert_eq!(n, 3);
        assert!(matches!(
            &out[0],
            WatcherEvent::Fs { resource, event } if *resource == ids[0] && *event == FsEvent::ContentChanged
        ));
        assert!(matches!(
            &out[1],
            WatcherEvent::Overflow {
                scope: OverflowScope::Global,
            }
        ));
        assert!(matches!(
            &out[2],
            WatcherEvent::Overflow {
                scope: OverflowScope::Resource(r),
            } if *r == ids[0]
        ));
        assert!(w.queued_events.is_empty());
        assert!(w.queued_overflow.is_empty());
    }

    #[test]
    fn drain_ready_with_only_overflow_drains_overflow() {
        // No `Fs` events queued — `drain_ready` still drains the overflow queue and reports a
        // non-zero count.
        let mut w = MockFsWatcher::new();
        w.inject_overflow(OverflowScope::Global);

        let mut out: Vec<WatcherEvent> = Vec::new();
        let n = w.drain_ready(&mut out).unwrap();
        assert_eq!(n, 1);
        assert!(matches!(
            &out[0],
            WatcherEvent::Overflow {
                scope: OverflowScope::Global,
            }
        ));
    }

    /// `as_fd()` is the edge-triggered readiness contract the bin's mio reactor relies on: the fd
    /// is not readable on a fresh mock, becomes readable after `inject` / `inject_overflow`, and is
    /// drained back to "not readable" by `drain_ready`. Pins the substrate so the bin's
    /// mio-registered drain loop sees the same readiness transitions a real kqueue / inotify fd
    /// would produce.
    ///
    /// Event-count assertions live in the `drain_ready_*` tests; this test asserts only on the fd's
    /// readability state.
    #[test]
    fn as_fd_becomes_readable_after_inject() {
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        // Before inject: read side is not readable.
        let mut sink = [0u8; 8];
        let n = nix::unistd::read(w.as_fd(), &mut sink);
        assert!(
            matches!(n, Err(nix::errno::Errno::EAGAIN)),
            "fresh mock has no pending readability; got {n:?}"
        );

        // After inject: read side is readable.
        w.inject(ids[0], FsEvent::ContentChanged);
        let n = nix::unistd::read(w.as_fd(), &mut sink)
            .expect("read side must be readable after inject");
        assert!(n >= 1, "inject must push at least one byte");

        // After drain_ready: substrate is drained even when the inject re-armed the readability
        // after the previous read.
        w.inject(ids[0], FsEvent::ContentChanged);
        let mut out = Vec::new();
        w.drain_ready(&mut out).unwrap();
        let n = nix::unistd::read(w.as_fd(), &mut sink);
        assert!(
            matches!(n, Err(nix::errno::Errno::EAGAIN)),
            "drain_ready must drain the readiness substrate; got {n:?}"
        );
    }

    // ---------------------------------------------------------------- registered_events

    /// `registered_events` reads the events on the latest `Watch` call. The engine emits a fresh
    /// `Watch` whenever `Resource.events_union` changes, so the latest call reflects the current
    /// per-Resource union.
    #[test]
    fn registered_events_returns_latest_watch_mask() {
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::File,
            ClassSet::CONTENT,
        )
        .unwrap();
        assert_eq!(w.registered_events(ids[0]), Some(ClassSet::CONTENT));

        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::File,
            ClassSet::CONTENT | ClassSet::METADATA,
        )
        .unwrap();
        assert_eq!(
            w.registered_events(ids[0]),
            Some(ClassSet::CONTENT | ClassSet::METADATA),
            "second Watch should overshadow the first"
        );
    }

    #[test]
    fn registered_events_returns_none_after_unwatch() {
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::Dir,
            ClassSet::STRUCTURE,
        )
        .unwrap();
        w.unwatch(ids[0]);
        assert_eq!(w.registered_events(ids[0]), None);
    }

    #[test]
    fn registered_events_returns_none_for_never_watched() {
        let ids = fresh_resource_ids(2);
        let mut w = MockFsWatcher::new();

        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::Unknown,
            ClassSet::EMPTY,
        )
        .unwrap();
        // ids[1] was never watched; registered_events should return None.
        assert_eq!(w.registered_events(ids[1]), None);
    }

    #[test]
    fn registered_events_after_rewatch_returns_new_mask_even_if_failed() {
        // The latest `Watch` op's events are returned regardless of whether the call succeeded —
        // `installed` tracks success, but the helper consumes the call log directly. This documents
        // the contract for tests that arm `fail_next_watch` to simulate FD-pressure on a re-register.
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::File,
            ClassSet::CONTENT,
        )
        .unwrap();
        w.fail_next_watch(WatchFailure::Pressure {
            errno: libc::EMFILE,
        });
        let _ = w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::File,
            ClassSet::CONTENT | ClassSet::METADATA,
        );
        // Helper reads the call log — sees the failed Watch attempt with the widened mask.
        assert_eq!(
            w.registered_events(ids[0]),
            Some(ClassSet::CONTENT | ClassSet::METADATA)
        );
    }

    // ---------------------------------------------------------------- MockProber

    fn fresh_profile_ids(n: usize) -> Vec<ProfileId> {
        let mut sm = SlotMap::<ProfileId, ()>::with_key();
        (0..n).map(|_| sm.insert(())).collect()
    }

    fn mk_req(profile: ProfileId, c: u64) -> ProbeRequest {
        ProbeRequest::AnchorFile {
            owner: ProbeOwner::Profile(profile),
            correlation: ProbeCorrelation::from(c),
            target_path: Arc::from(PathBuf::from("/dev/null")),
        }
    }

    #[test]
    fn mock_prober_records_submit() {
        let pids = fresh_profile_ids(1);
        let mp = MockProber::new();

        mp.submit(mk_req(pids[0], 1));
        let drained = mp.take_submitted();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].owner(), ProbeOwner::Profile(pids[0]));
        assert_eq!(drained[0].correlation(), ProbeCorrelation::from(1));
    }

    #[test]
    fn mock_prober_records_cancel() {
        let pids = fresh_profile_ids(1);
        let mp = MockProber::new();

        mp.cancel(ProbeOwner::Profile(pids[0]));
        let drained = mp.take_cancelled();
        assert_eq!(drained, vec![ProbeOwner::Profile(pids[0])]);
    }

    #[test]
    fn mock_prober_take_drains_to_empty() {
        let pids = fresh_profile_ids(2);
        let mp = MockProber::new();

        mp.submit(mk_req(pids[0], 1));
        mp.submit(mk_req(pids[1], 2));
        assert_eq!(mp.take_submitted().len(), 2);
        // Second take is empty.
        assert!(mp.take_submitted().is_empty());
    }

    #[test]
    fn mock_prober_thread_safe_concurrent_submit() {
        let mp = Arc::new(MockProber::new());
        let pids = fresh_profile_ids(15);
        let chunks: Vec<Vec<ProfileId>> = pids.chunks(5).map(<[ProfileId]>::to_vec).collect();

        let mut threads = Vec::new();
        for chunk in chunks {
            let mp = Arc::clone(&mp);
            threads.push(std::thread::spawn(move || {
                for (i, p) in chunk.into_iter().enumerate() {
                    mp.submit(mk_req(p, i as u64));
                }
            }));
        }
        for t in threads {
            t.join().unwrap();
        }
        assert_eq!(mp.take_submitted().len(), 15);
    }

    #[test]
    fn mock_prober_impls_send_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<MockProber>();
    }
}
