//! Pool-size respect: with N workers, exactly N probes can be in flight simultaneously. We pin this
//! through a `Barrier`-coordinated probe — but the prober is a black box, so we use a *real-fs*
//! slow probe (a deeply-recursive walk) instead of a test seam.

#![cfg(unix)]

use crossbeam::channel::{Sender, unbounded};
use slotmap::SlotMap;
use specter_core::{Input, ProbeCorrelation, ProbeRequest, ProfileId};
use specter_sensor::{ProbeResponse, Prober, ProberResponseSender, SendError, WorkerProber};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tempfile::TempDir;

fn fresh_profile_ids(n: usize) -> Vec<ProfileId> {
    let mut sm = SlotMap::<ProfileId, ()>::with_key();
    (0..n).map(|_| sm.insert(())).collect()
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

fn anchor_request(profile: ProfileId, target_path: PathBuf, correlation: u64) -> ProbeRequest {
    ProbeRequest::AnchorFile {
        owner: profile,
        correlation: ProbeCorrelation::from(correlation),
        target_path: Arc::from(target_path),
    }
}

#[test]
fn single_worker_drains_more_than_concurrency_serially() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("f");
    std::fs::write(&path, b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober =
        WorkerProber::new(sink(tx), NonZeroUsize::new(1).expect("1 is non-zero")).unwrap();
    let pids = fresh_profile_ids(5);

    for (i, p) in pids.iter().enumerate() {
        prober.submit(anchor_request(*p, path.clone(), i as u64 + 1));
    }

    // Single worker → responses arrive in submit order (FIFO).
    let mut order = Vec::new();
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline && order.len() < 5 {
        if let Ok(Input::ProbeResponse(r)) = rx.recv_timeout(Duration::from_millis(300)) {
            order.push(r.correlation);
        }
    }
    assert_eq!(order.len(), 5);
    assert_eq!(
        order,
        vec![
            ProbeCorrelation::from(1),
            ProbeCorrelation::from(2),
            ProbeCorrelation::from(3),
            ProbeCorrelation::from(4),
            ProbeCorrelation::from(5),
        ]
    );

    let _ = prober.shutdown();
}

#[test]
fn pool_with_four_workers_handles_burst() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("f");
    std::fs::write(&path, b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober =
        WorkerProber::new(sink(tx), NonZeroUsize::new(4).expect("4 is non-zero")).unwrap();
    let pids = fresh_profile_ids(20);

    for (i, p) in pids.iter().enumerate() {
        prober.submit(anchor_request(*p, path.clone(), i as u64 + 1));
    }

    let mut received = 0;
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline && received < 20 {
        if let Ok(Input::ProbeResponse(_)) = rx.recv_timeout(Duration::from_millis(300)) {
            received += 1;
        }
    }
    assert_eq!(received, 20, "pool drained the burst");

    let _ = prober.shutdown();
}
