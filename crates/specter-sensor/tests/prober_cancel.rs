//! Cancellation contract: `cancel(profile)` is best-effort. The
//! deterministic cases (cancel-without-submit; resubmit-after-cancel
//! runs; cancel-after-completion is a no-op) live here. The
//! racy-by-construction "cancel skips a queued probe" case is covered
//! by sibling unit tests via `run_worker` with a hand-seeded
//! expectation map — that's where determinism is reachable.

#![cfg(unix)]

use crossbeam::channel::unbounded;
use slotmap::SlotMap;
use specter_core::{
    Input, ProbeCorrelation, ProbeKind, ProbeRequest, ProbeResult, ProfileId, ResourceId,
    ScanConfig,
};
use specter_sensor::{Prober, WorkerProber};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;

fn fresh_profile_id() -> ProfileId {
    let mut sm = SlotMap::<ProfileId, ()>::with_key();
    sm.insert(())
}

fn mk_request(profile: ProfileId, anchor: PathBuf, correlation: u64) -> ProbeRequest {
    ProbeRequest {
        profile,
        correlation: ProbeCorrelation(correlation),
        kind: ProbeKind::File,
        target_resource: ResourceId::default(),
        target_path: anchor,
        scan_config: ScanConfig::builder().build(),
        captured_with: 0,
        baseline_subtree: None,
        force_walk: BTreeSet::new(),
        forced: false,
    }
}

#[test]
fn cancel_without_prior_submit_is_noop() {
    let (tx, _rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&tx, 1).unwrap();
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
    let prober = WorkerProber::new(&tx, 1).unwrap();
    let p = fresh_profile_id();

    prober.submit(mk_request(p, path, 1));
    let resp = match rx.recv_timeout(Duration::from_secs(2)).expect("response") {
        Input::ProbeResponse(r) => r,
        other => panic!("unexpected: {other:?}"),
    };
    assert!(matches!(resp.result, ProbeResult::Ok(_)));

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
    let prober = WorkerProber::new(&tx, 1).unwrap();
    let p = fresh_profile_id();

    prober.submit(mk_request(p, path.clone(), 1));
    prober.cancel(p);
    prober.submit(mk_request(p, path, 2));

    // c1 may or may not arrive (race). c2 must arrive.
    let mut got_c2 = false;
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline && !got_c2 {
        if let Ok(Input::ProbeResponse(r)) = rx.recv_timeout(Duration::from_millis(200))
            && r.correlation == ProbeCorrelation(2)
        {
            got_c2 = true;
        }
    }
    assert!(got_c2, "c2 must arrive after resubmit");

    let _ = prober.shutdown();
}
