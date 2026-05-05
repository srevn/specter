//! Pure-Rust `FsWatcher` test fixture.
//!
//! Records every mutator call into a `Vec<WatcherCall>`; tests inject
//! events for `poll_until` to deliver synchronously; the wake handle
//! increments a shared counter so cross-thread wake patterns are
//! observable.
//!
//! No FFI, no kqueue, no platform gates — compiles on every target
//! (Linux CI, macOS dev, FreeBSD prod). Production builds opt out via
//! the `testkit` Cargo feature; consumers attach it under
//! `[dev-dependencies]`.

use crate::{FsWatcher, Prober, WakeHandle};
use slotmap::SecondaryMap;
use specter_core::{ClassSet, FsEvent, ProbeRequest, ProfileId, ResourceId, WatchOpts};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

/// One recorded call into the watcher's mutator surface.
#[derive(Debug, Clone)]
pub enum WatcherCall {
    Watch {
        resource: ResourceId,
        path: PathBuf,
        opts: WatchOpts,
    },
    Unwatch {
        resource: ResourceId,
    },
    Suppress {
        resource: ResourceId,
    },
    Unsuppress {
        resource: ResourceId,
    },
}

#[derive(Debug, Default)]
pub struct MockFsWatcher {
    /// Append-only log of every mutator call. Ordering is the order of
    /// invocation — useful for asserting cross-step sequences.
    pub calls: Vec<WatcherCall>,
    /// Current installed-watch state: `r ∈ installed` ⇔ last `watch(r)`
    /// succeeded and no subsequent `unwatch(r)` cleared it.
    pub installed: SecondaryMap<ResourceId, PathBuf>,
    /// Current suppress state: present ⇔ last edge was `suppress`.
    pub suppressed: SecondaryMap<ResourceId, ()>,
    /// Events queued for delivery on the next `poll_until` call.
    pub queued_events: Vec<(ResourceId, FsEvent)>,
    /// Set by `fail_next_watch` to simulate FD pressure / EACCES /
    /// ENOENT — the **next** `watch()` call returns `Err(errno)`
    /// without modifying state. One-shot; consumed on read.
    pub next_watch_errno: Option<i32>,
    /// Wake-counter shared with every cloned `MockWakeHandle`.
    pub waker: Arc<MockWaker>,
}

/// Shared counter for cross-thread wake observation. Cloned wake
/// handles all bump the same `Mutex<u32>`; tests assert the running
/// total after spawning their wakers.
#[derive(Debug, Default)]
pub struct MockWaker {
    pub woken: Mutex<u32>,
}

impl MockFsWatcher {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Queue an event for delivery on the next `poll_until` call. The
    /// queue drains entirely on poll — caller is responsible for
    /// re-injecting between successive polls.
    pub fn inject(&mut self, r: ResourceId, ev: FsEvent) {
        self.queued_events.push((r, ev));
    }

    /// Cause the next `watch` call to fail with `errno`. Consumed on
    /// read — subsequent watches succeed unless re-armed.
    pub const fn fail_next_watch(&mut self, errno: i32) {
        self.next_watch_errno = Some(errno);
    }

    /// Currently-registered event-class mask for `r`, derived from the
    /// last lifecycle op in `calls`:
    ///
    /// - Returns `Some(events)` if the most recent op for `r` is a
    ///   `Watch` (regardless of whether that `Watch` succeeded — see
    ///   [`fail_next_watch`](Self::fail_next_watch)).
    /// - Returns `None` if the most recent op is `Unwatch` or if `r` has
    ///   no lifecycle ops in the call log.
    ///
    /// `Suppress` / `Unsuppress` ops do not affect the mask and are
    /// skipped during the walk. Under R2 / D11 the engine emits a fresh
    /// `WatchOp::Watch` whenever `Resource.events_union` widens or
    /// narrows, so the latest `Watch` reflects the engine's current
    /// per-Resource union.
    #[must_use]
    pub fn registered_events(&self, r: ResourceId) -> Option<ClassSet> {
        for c in self.calls.iter().rev() {
            match c {
                WatcherCall::Watch { resource, opts, .. } if *resource == r => {
                    return Some(opts.events);
                }
                WatcherCall::Unwatch { resource } if *resource == r => return None,
                _ => {}
            }
        }
        None
    }
}

impl FsWatcher for MockFsWatcher {
    fn watch(&mut self, r: ResourceId, path: &Path, opts: WatchOpts) -> io::Result<()> {
        self.calls.push(WatcherCall::Watch {
            resource: r,
            path: path.to_owned(),
            opts,
        });
        if let Some(errno) = self.next_watch_errno.take() {
            return Err(io::Error::from_raw_os_error(errno));
        }
        self.installed.insert(r, path.to_owned());
        Ok(())
    }

    fn unwatch(&mut self, r: ResourceId) {
        self.calls.push(WatcherCall::Unwatch { resource: r });
        self.installed.remove(r);
        self.suppressed.remove(r);
    }

    fn suppress(&mut self, r: ResourceId) {
        self.calls.push(WatcherCall::Suppress { resource: r });
        if self.installed.contains_key(r) {
            self.suppressed.insert(r, ());
        }
    }

    fn unsuppress(&mut self, r: ResourceId) {
        self.calls.push(WatcherCall::Unsuppress { resource: r });
        self.suppressed.remove(r);
    }

    fn poll_until(
        &mut self,
        _deadline: Option<Instant>,
        out: &mut Vec<(ResourceId, FsEvent)>,
    ) -> io::Result<usize> {
        let drained = std::mem::take(&mut self.queued_events);
        let n = drained.len();
        out.extend(drained);
        Ok(n)
    }

    fn wake_handle(&self) -> Box<dyn WakeHandle> {
        Box::new(MockWakeHandle {
            waker: Arc::clone(&self.waker),
        })
    }
}

#[derive(Debug, Clone)]
struct MockWakeHandle {
    waker: Arc<MockWaker>,
}

impl WakeHandle for MockWakeHandle {
    fn wake(&self) {
        let mut count = self.waker.woken.lock().expect("MockWaker poisoned");
        *count += 1;
    }

    fn clone_box(&self) -> Box<dyn WakeHandle> {
        Box::new(self.clone())
    }
}

/// Pure-Rust [`Prober`] test fixture.
///
/// Records every `submit` and `cancel` call into `Mutex<Vec<...>>`s
/// for assertion. Does *not* synthesize `ProbeResponse`s — engine
/// tests inject responses via `core::testkit::sensor::MockSensor`'s
/// `probe_response` constructor directly.
///
/// `MockProber` is the right fixture for *bin / integration* tests
/// that need to assert "the engine emitted these probe operations to
/// the prober" without spinning up a real `WorkerProber` against a
/// `tempfile::TempDir`. Real-fs round-trip tests use `WorkerProber`.
#[derive(Debug, Default)]
pub struct MockProber {
    pub submitted: Mutex<Vec<ProbeRequest>>,
    pub cancelled: Mutex<Vec<ProfileId>>,
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
    pub fn take_cancelled(&self) -> Vec<ProfileId> {
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

    fn cancel(&self, profile: ProfileId) {
        self.cancelled
            .lock()
            .expect("MockProber poisoned")
            .push(profile);
    }
}

#[cfg(test)]
mod tests {
    use super::{MockFsWatcher, MockProber, WatcherCall};
    use crate::{FsWatcher, Prober, WakeHandle};
    use slotmap::SlotMap;
    use specter_core::{
        ClassSet, FsEvent, ProbeCorrelation, ProbeKind, ProbeRequest, ProfileId, ResourceId,
        ScanConfig, WatchOpts,
    };
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

        w.watch(ids[0], &PathBuf::from("/tmp/a"), WatchOpts::default())
            .expect("watch ok");

        assert_eq!(w.calls.len(), 1);
        assert!(matches!(
            &w.calls[0],
            WatcherCall::Watch { resource, .. } if *resource == ids[0]
        ));
        assert_eq!(w.installed.get(ids[0]).unwrap(), &PathBuf::from("/tmp/a"));
    }

    #[test]
    fn watch_returns_error_when_armed() {
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();
        w.fail_next_watch(libc::EMFILE);

        let res = w.watch(ids[0], &PathBuf::from("/tmp/a"), WatchOpts::default());
        assert!(res.is_err());
        assert_eq!(res.err().unwrap().raw_os_error(), Some(libc::EMFILE));
        // The call is still recorded; the error short-circuits installation.
        assert_eq!(w.calls.len(), 1);
        assert!(!w.installed.contains_key(ids[0]));

        // One-shot — the next watch succeeds.
        w.watch(ids[0], &PathBuf::from("/tmp/a"), WatchOpts::default())
            .expect("second watch ok");
        assert!(w.installed.contains_key(ids[0]));
    }

    #[test]
    fn unwatch_clears_installed_and_suppressed() {
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.watch(ids[0], &PathBuf::from("/tmp/a"), WatchOpts::default())
            .unwrap();
        w.suppress(ids[0]);
        assert!(w.suppressed.contains_key(ids[0]));

        w.unwatch(ids[0]);
        assert!(!w.installed.contains_key(ids[0]));
        assert!(!w.suppressed.contains_key(ids[0]));
    }

    #[test]
    fn poll_until_drains_queued_events() {
        let ids = fresh_resource_ids(2);
        let mut w = MockFsWatcher::new();

        w.inject(ids[0], FsEvent::Modified);
        w.inject(ids[1], FsEvent::Renamed);

        let mut out = Vec::new();
        let n = w.poll_until(None, &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out.len(), 2);
        assert!(w.queued_events.is_empty());

        // Second poll: nothing queued.
        out.clear();
        let n = w.poll_until(None, &mut out).unwrap();
        assert_eq!(n, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn wake_handle_increments_counter_across_threads() {
        let w = MockFsWatcher::new();
        let handle = w.wake_handle();
        let waker = Arc::clone(&w.waker);

        let mut threads = Vec::new();
        for _ in 0..3 {
            let h: Box<dyn WakeHandle> = handle.clone();
            threads.push(std::thread::spawn(move || h.wake()));
        }
        for t in threads {
            t.join().unwrap();
        }

        assert_eq!(*waker.woken.lock().unwrap(), 3);
    }

    #[test]
    fn suppress_unsuppress_records_calls() {
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.watch(ids[0], &PathBuf::from("/tmp/a"), WatchOpts::default())
            .unwrap();
        w.suppress(ids[0]);
        w.unsuppress(ids[0]);

        let kinds: Vec<&str> = w
            .calls
            .iter()
            .map(|c| match c {
                WatcherCall::Watch { .. } => "watch",
                WatcherCall::Unwatch { .. } => "unwatch",
                WatcherCall::Suppress { .. } => "suppress",
                WatcherCall::Unsuppress { .. } => "unsuppress",
            })
            .collect();
        assert_eq!(kinds, vec!["watch", "suppress", "unsuppress"]);
        assert!(!w.suppressed.contains_key(ids[0])); // Net: unsuppressed.
    }

    #[test]
    fn suppress_on_unwatched_does_not_install_state() {
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        // No prior `watch(ids[0])`; suppress should be a no-op on state.
        w.suppress(ids[0]);
        assert_eq!(w.calls.len(), 1);
        assert!(!w.suppressed.contains_key(ids[0]));
    }

    // ---------------------------------------------------------------- registered_events

    /// `registered_events` reads the events on the latest `Watch` call.
    /// Under R2 / D11 the engine emits a fresh `Watch` whenever
    /// `Resource.events_union` changes, so the latest call reflects the
    /// current per-Resource union.
    #[test]
    fn registered_events_returns_latest_watch_mask() {
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        let opts1 = WatchOpts {
            events: ClassSet::CONTENT,
            ..WatchOpts::default()
        };
        let opts2 = WatchOpts {
            events: ClassSet::CONTENT | ClassSet::METADATA,
            ..WatchOpts::default()
        };

        w.watch(ids[0], &PathBuf::from("/tmp/a"), opts1).unwrap();
        assert_eq!(w.registered_events(ids[0]), Some(ClassSet::CONTENT));

        w.watch(ids[0], &PathBuf::from("/tmp/a"), opts2).unwrap();
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
            WatchOpts {
                events: ClassSet::STRUCTURE,
                ..WatchOpts::default()
            },
        )
        .unwrap();
        w.unwatch(ids[0]);
        assert_eq!(w.registered_events(ids[0]), None);
    }

    #[test]
    fn registered_events_returns_none_for_never_watched() {
        let ids = fresh_resource_ids(2);
        let mut w = MockFsWatcher::new();

        w.watch(ids[0], &PathBuf::from("/tmp/a"), WatchOpts::default())
            .unwrap();
        // ids[1] was never watched; registered_events should return None.
        assert_eq!(w.registered_events(ids[1]), None);
    }

    #[test]
    fn registered_events_skips_suppress_unsuppress_ops() {
        // Suppress / Unsuppress don't change the events mask; the helper
        // walks past them to find the latest Watch.
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            WatchOpts {
                events: ClassSet::CONTENT,
                ..WatchOpts::default()
            },
        )
        .unwrap();
        w.suppress(ids[0]);
        w.unsuppress(ids[0]);
        assert_eq!(w.registered_events(ids[0]), Some(ClassSet::CONTENT));
    }

    #[test]
    fn registered_events_after_rewatch_returns_new_mask_even_if_failed() {
        // The latest `Watch` op's events are returned regardless of
        // whether the call succeeded — `installed` tracks success, but
        // the helper consumes the call log directly. This documents the
        // contract for tests that arm `fail_next_watch` to simulate
        // FD-pressure on a re-register.
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            WatchOpts {
                events: ClassSet::CONTENT,
                ..WatchOpts::default()
            },
        )
        .unwrap();
        w.fail_next_watch(libc::EMFILE);
        let _ = w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            WatchOpts {
                events: ClassSet::CONTENT | ClassSet::METADATA,
                ..WatchOpts::default()
            },
        );
        // Helper reads the call log — sees the failed Watch attempt
        // with the widened mask.
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
        ProbeRequest {
            profile,
            correlation: ProbeCorrelation(c),
            kind: ProbeKind::File,
            target_resource: specter_core::ResourceId::default(),
            target_path: PathBuf::from("/dev/null"),
            scan_config: ScanConfig::builder().build(),
            captured_with: 0,
            baseline_subtree: None,
            force_walk: std::collections::BTreeSet::new(),
            forced: false,
        }
    }

    #[test]
    fn mock_prober_records_submit() {
        let pids = fresh_profile_ids(1);
        let mp = MockProber::new();

        mp.submit(mk_req(pids[0], 1));
        let drained = mp.take_submitted();
        assert_eq!(drained.len(), 1);
        assert_eq!(drained[0].profile, pids[0]);
        assert_eq!(drained[0].correlation, ProbeCorrelation(1));
    }

    #[test]
    fn mock_prober_records_cancel() {
        let pids = fresh_profile_ids(1);
        let mp = MockProber::new();

        mp.cancel(pids[0]);
        let drained = mp.take_cancelled();
        assert_eq!(drained, vec![pids[0]]);
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
