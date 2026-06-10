//! `shutdown(self)` cleanly drops the queue Sender, joins workers, and returns each worker's
//! `(index, thread::Result<()>)` for the bin to log. `drop(prober)` without an explicit shutdown
//! still terminates workers via channel disconnect — workers run to completion on pending probes,
//! then exit.

#![cfg(unix)]

use crossbeam::channel::{Sender, unbounded};
use slotmap::SlotMap;
use specter_core::{Input, ProbeCorrelation, ProbeRequest, ProfileId};
use specter_sensor::{ProbeResponse, Prober, ProberResponseSender, SendError, WorkerProber};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

fn fresh_profile_id() -> ProfileId {
    let mut sm = SlotMap::<ProfileId, ()>::with_key();
    sm.insert(())
}

const fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("non-zero literal in test fixture")
}

/// Mirror of the bin's `DriverProberSender` — wraps a single `Sender<Input>` clone and rewraps each
/// `ProbeResponse` as `Input::ProbeResponse(_)` on the wire.
struct TestProberSink {
    tx: Sender<Input>,
}

impl ProberResponseSender for TestProberSink {
    fn send(&self, response: ProbeResponse) -> Result<(), SendError> {
        self.tx
            .send(Input::ProbeResponse(response))
            .map_err(|_| SendError::Disconnected)
    }
}

fn sink(tx: Sender<Input>) -> Arc<dyn ProberResponseSender> {
    Arc::new(TestProberSink { tx })
}

fn mk_request(profile: ProfileId, target_path: PathBuf, correlation: u64) -> ProbeRequest {
    ProbeRequest::AnchorFile {
        owner: profile,
        correlation: ProbeCorrelation::from(correlation),
        target_path: Arc::from(target_path),
    }
}

#[test]
fn shutdown_with_no_pending_probes_returns_ok_per_worker() {
    let (tx, _rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), nz(4)).unwrap();
    let results = prober.shutdown();
    assert_eq!(results.len(), 4);
    for (i, r) in results {
        r.unwrap_or_else(|_| panic!("worker {i} exited unclean"));
    }
}

#[test]
fn shutdown_after_completed_probe_returns_ok() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("f");
    std::fs::write(&path, b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), nz(2)).unwrap();
    let p = fresh_profile_id();
    prober.submit(mk_request(p, path, 1));
    let _ = rx.recv_timeout(Duration::from_secs(2)).expect("response");
    let results = prober.shutdown();
    assert_eq!(results.len(), 2);
    for (i, r) in results {
        r.unwrap_or_else(|_| panic!("worker {i} exited unclean"));
    }
}

#[test]
fn drop_without_shutdown_terminates_workers_via_channel_disconnect() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("f");
    std::fs::write(&path, b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    {
        let prober = WorkerProber::new(sink(tx), nz(1)).unwrap();
        let p = fresh_profile_id();
        prober.submit(mk_request(p, path, 1));
        // Wait for response before dropping — workers process the pending probe before the queue
        // Sender drop reaches them.
        let _ = rx.recv_timeout(Duration::from_secs(2)).expect("response");
        // Implicit drop here: queue_tx drops with prober → workers see Disconnected on next recv →
        // exit on their own. We don't observe their JoinHandle (detached).
    }
    // After drop, the prober's `Arc<dyn ProberResponseSender>` clones have all dropped (workers
    // exited), releasing the only `Sender<Input>` clone inside the sink. The receiver observes
    // `Disconnected`; either way, no spurious post-drop messages.
    assert!(rx.try_recv().is_err(), "no spurious post-drop messages");
}
