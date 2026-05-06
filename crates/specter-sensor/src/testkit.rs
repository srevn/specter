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

use crate::{FsWatcher, OverflowScope, Prober, WakeHandle, WatchFailure, WatcherEvent};
use slotmap::SecondaryMap;
use specter_core::{ClassSet, FsEvent, ProbeRequest, ProfileId, ResourceId, ResourceKind};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use std::time::Instant;

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
    Suppress {
        resource: ResourceId,
    },
    Unsuppress {
        resource: ResourceId,
    },
}

/// Live record of an installed watch on the mock. One entry per
/// successfully-`watch()`ed resource; cleared on `unwatch()`. Tests
/// inspect this for the post-install state without walking the call
/// log.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct MockEntry {
    pub path: PathBuf,
    pub kind: ResourceKind,
    pub events: ClassSet,
}

#[derive(Debug, Default)]
pub struct MockFsWatcher {
    /// Append-only log of every mutator call. Ordering is the order of
    /// invocation — useful for asserting cross-step sequences.
    pub calls: Vec<WatcherCall>,
    /// Current installed-watch state: `r ∈ installed` ⇔ last `watch(r)`
    /// succeeded and no subsequent `unwatch(r)` cleared it.
    pub installed: SecondaryMap<ResourceId, MockEntry>,
    /// Current suppress state: present ⇔ last edge was `suppress`.
    pub suppressed: SecondaryMap<ResourceId, ()>,
    /// Per-resource events queued for delivery on the next `poll_until`
    /// call. Drained into [`WatcherEvent::Fs`] before any queued
    /// overflow scopes (preserves the natural "events first, kernel
    /// signals last" ordering the bin's loop expects).
    pub queued_events: Vec<(ResourceId, FsEvent)>,
    /// Overflow scopes queued for delivery on the next `poll_until`
    /// call. Drained into [`WatcherEvent::Overflow`] **after** the
    /// per-resource events. Tests use this to exercise the bin's
    /// overflow-routing path without wiring a real inotify backend.
    pub queued_overflow: Vec<OverflowScope>,
    /// Set by `fail_next_watch` to simulate FD pressure / kind
    /// mismatch / programmer error — the **next** `watch()` call returns
    /// `Err(failure)` without modifying state. One-shot; consumed on read.
    pub next_watch_failure: Option<WatchFailure>,
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

    /// Queue a per-resource event for delivery on the next `poll_until`
    /// call. The queue drains entirely on poll — caller is responsible
    /// for re-injecting between successive polls.
    pub fn inject(&mut self, r: ResourceId, ev: FsEvent) {
        self.queued_events.push((r, ev));
    }

    /// Queue an overflow signal for delivery on the next `poll_until`
    /// call. Drained into [`WatcherEvent::Overflow`] **after** any
    /// queued per-resource events on the same poll, mirroring the
    /// inotify backend's natural ordering (`Fs` events for the records
    /// preceding the overflow marker; `Overflow` once the kernel signals
    /// the queue ran dry).
    ///
    /// Useful for engine / bin tests that need to assert the `Overflow`
    /// routing path (Phase B11) without spinning up a real inotify
    /// instance and stressing it past `max_queued_events`.
    pub fn inject_overflow(&mut self, scope: OverflowScope) {
        self.queued_overflow.push(scope);
    }

    /// Cause the next `watch` call to fail with `failure`. Consumed on
    /// read — subsequent watches succeed unless re-armed.
    pub const fn fail_next_watch(&mut self, failure: WatchFailure) {
        self.next_watch_failure = Some(failure);
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
        out: &mut Vec<WatcherEvent>,
    ) -> Result<usize, WatchFailure> {
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
    use crate::{FsWatcher, OverflowScope, Prober, WakeHandle, WatchFailure, WatcherEvent};
    use slotmap::SlotMap;
    use specter_core::{
        ClassSet, FsEvent, ProbeCorrelation, ProbeKind, ProbeRequest, ProfileId, ResourceId,
        ResourceKind, ScanConfig,
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
    fn unwatch_clears_installed_and_suppressed() {
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::Unknown,
            ClassSet::EMPTY,
        )
        .unwrap();
        w.suppress(ids[0]);
        assert!(w.suppressed.contains_key(ids[0]));

        w.unwatch(ids[0]);
        assert!(!w.installed.contains_key(ids[0]));
        assert!(!w.suppressed.contains_key(ids[0]));
    }

    #[test]
    fn poll_until_drains_queued_events_as_fs_variant() {
        let ids = fresh_resource_ids(2);
        let mut w = MockFsWatcher::new();

        w.inject(ids[0], FsEvent::Modified);
        w.inject(ids[1], FsEvent::Renamed);

        let mut out: Vec<WatcherEvent> = Vec::new();
        let n = w.poll_until(None, &mut out).unwrap();
        assert_eq!(n, 2);
        assert_eq!(out.len(), 2);
        assert!(w.queued_events.is_empty());
        assert!(matches!(
            &out[0],
            WatcherEvent::Fs { resource, event } if *resource == ids[0] && *event == FsEvent::Modified
        ));
        assert!(matches!(
            &out[1],
            WatcherEvent::Fs { resource, event } if *resource == ids[1] && *event == FsEvent::Renamed
        ));

        // Second poll: nothing queued.
        out.clear();
        let n = w.poll_until(None, &mut out).unwrap();
        assert_eq!(n, 0);
        assert!(out.is_empty());
    }

    #[test]
    fn poll_until_drains_queued_overflow_after_fs_events() {
        // Ordering invariant: any per-resource events queued before the
        // poll surface as `Fs` records first, then the queued overflow
        // scopes as `Overflow` records. This mirrors the inotify
        // backend's natural drain order — events for records preceding
        // the overflow marker come first, then `IN_Q_OVERFLOW`.
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.inject(ids[0], FsEvent::Modified);
        w.inject_overflow(OverflowScope::Global);
        w.inject_overflow(OverflowScope::Resource(ids[0]));

        let mut out: Vec<WatcherEvent> = Vec::new();
        let n = w.poll_until(None, &mut out).unwrap();
        assert_eq!(n, 3);
        assert!(matches!(
            &out[0],
            WatcherEvent::Fs { resource, event } if *resource == ids[0] && *event == FsEvent::Modified
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
    fn poll_until_with_only_overflow_drains_overflow() {
        // No `Fs` events queued — `poll_until` still drains the
        // overflow queue and reports a non-zero count.
        let mut w = MockFsWatcher::new();
        w.inject_overflow(OverflowScope::Global);

        let mut out: Vec<WatcherEvent> = Vec::new();
        let n = w.poll_until(None, &mut out).unwrap();
        assert_eq!(n, 1);
        assert!(matches!(
            &out[0],
            WatcherEvent::Overflow {
                scope: OverflowScope::Global,
            }
        ));
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

        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::Unknown,
            ClassSet::EMPTY,
        )
        .unwrap();
        w.suppress(ids[0]);
        w.unsuppress(ids[0]);

        let labels: Vec<&str> = w
            .calls
            .iter()
            .map(|c| match c {
                WatcherCall::Watch { .. } => "watch",
                WatcherCall::Unwatch { .. } => "unwatch",
                WatcherCall::Suppress { .. } => "suppress",
                WatcherCall::Unsuppress { .. } => "unsuppress",
            })
            .collect();
        assert_eq!(labels, vec!["watch", "suppress", "unsuppress"]);
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
    fn registered_events_skips_suppress_unsuppress_ops() {
        // Suppress / Unsuppress don't change the events mask; the helper
        // walks past them to find the latest Watch.
        let ids = fresh_resource_ids(1);
        let mut w = MockFsWatcher::new();

        w.watch(
            ids[0],
            &PathBuf::from("/tmp/a"),
            ResourceKind::File,
            ClassSet::CONTENT,
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
