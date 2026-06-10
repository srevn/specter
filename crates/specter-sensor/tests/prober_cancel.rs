//! Cancellation contract: `cancel(profile)` is best-effort. The deterministic cases
//! (cancel-without-submit; resubmit-after-cancel runs; cancel-after-completion is a no-op) live here.
//! The racy-by-construction "cancel skips a queued probe" case is covered by sibling unit tests via
//! `run_worker` with a hand-seeded expectation map — that's where determinism is reachable.

#![cfg(unix)]

use crossbeam::channel::{Sender, unbounded};
use slotmap::SlotMap;
use specter_core::{Input, ProbeCorrelation, ProbeOutcome, ProbeRequest, ProfileId};
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
fn cancel_without_prior_submit_is_noop() {
    let (tx, _rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), nz(1)).unwrap();
    let p = fresh_profile_id();

    // No panic; subsequent submits unaffected.
    prober.cancel(p);

    let _ = prober.shutdown();
}

#[test]
fn cancel_after_completion_is_noop() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("f");
    std::fs::write(&path, b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), nz(1)).unwrap();
    let p = fresh_profile_id();

    prober.submit(mk_request(p, path, 1));
    let resp = match rx.recv_timeout(Duration::from_secs(2)).expect("response") {
        Input::ProbeResponse(r) => r,
        other => panic!("unexpected: {other:?}"),
    };
    assert!(matches!(resp.outcome, ProbeOutcome::AnchorOk(_)));

    // Cancel after completion: no panic; no spurious second response.
    prober.cancel(p);
    assert!(rx.recv_timeout(Duration::from_millis(100)).is_err());

    let _ = prober.shutdown();
}

#[test]
fn resubmit_after_cancel_runs_with_new_correlation() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("f");
    std::fs::write(&path, b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), nz(1)).unwrap();
    let p = fresh_profile_id();

    prober.submit(mk_request(p, path.clone(), 1));
    prober.cancel(p);
    prober.submit(mk_request(p, path, 2));

    // c1 may or may not arrive (race). c2 must arrive.
    let mut got_c2 = false;
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline && !got_c2 {
        if let Ok(Input::ProbeResponse(r)) = rx.recv_timeout(Duration::from_millis(200))
            && r.correlation == ProbeCorrelation::from(2)
        {
            got_c2 = true;
        }
    }
    assert!(got_c2, "c2 must arrive after resubmit");

    let _ = prober.shutdown();
}
