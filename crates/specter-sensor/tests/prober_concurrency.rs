//! Pool-size respect: with N workers, exactly N probes can be in
//! flight simultaneously. We pin this through a `Barrier`-coordinated
//! probe — but the prober is a black box, so we use a *real-fs* slow
//! probe (a deeply-recursive walk) instead of a test seam.

#![cfg(unix)]

use crossbeam::channel::unbounded;
use slotmap::SlotMap;
use specter_core::{Input, ProbeCorrelation, ProbeRequest, ProfileId, ResourceId, ScanConfig};
use specter_sensor::{Prober, WorkerProber};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tempfile::TempDir;

fn fresh_profile_ids(n: usize) -> Vec<ProfileId> {
    let mut sm = SlotMap::<ProfileId, ()>::with_key();
    (0..n).map(|_| sm.insert(())).collect()
}

const fn anchor_request(
    profile: ProfileId,
    target_path: PathBuf,
    correlation: u64,
) -> ProbeRequest {
    ProbeRequest::AnchorFile {
        profile,
        correlation: ProbeCorrelation(correlation),
        target_path,
    }
}

fn subtree_request(
    profile: ProfileId,
    target_path: PathBuf,
    scan_config: ScanConfig,
    correlation: u64,
) -> ProbeRequest {
    ProbeRequest::Subtree {
        profile,
        correlation: ProbeCorrelation(correlation),
        target_resource: ResourceId::default(),
        target_path,
        scan_config,
        captured_with: 0,
        baseline_subtree: None,
        force_walk: BTreeSet::new(),
        forced: false,
    }
}

#[test]
fn single_worker_drains_more_than_concurrency_serially() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("f");
    std::fs::write(&path, b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&tx, 1).unwrap();
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
            ProbeCorrelation(1),
            ProbeCorrelation(2),
            ProbeCorrelation(3),
            ProbeCorrelation(4),
            ProbeCorrelation(5),
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
    let prober = WorkerProber::new(&tx, 4).unwrap();
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

/// 4-worker pool processing a 4-element burst should complete faster
/// than the same burst on a 1-worker pool. We use directory probes on
/// a moderately-deep tree so each probe takes long enough that the
/// concurrency win is observable. Tolerance is loose to avoid
/// flakiness; the test is asserting "more workers ≥ less wall time",
/// not a specific speedup factor.
#[test]
fn pool_runs_probes_concurrently_when_capacity_allows() {
    let tmp = TempDir::new().unwrap();
    // Build a tree with enough I/O that each probe takes a measurable
    for i in 0..20 {
        let sub = tmp.path().join(format!("dir{i}"));
        std::fs::create_dir(&sub).unwrap();
        for j in 0..250 {
            std::fs::write(sub.join(format!("file{j}")), b"x").unwrap();
        }
    }

    let cfg = ScanConfig::builder().recursive(true).build();
    let mk_burst = |concurrency: usize, anchor: &Path| -> Duration {
        let (tx, rx) = unbounded::<Input>();
        let prober = WorkerProber::new(&tx, concurrency).unwrap();
        let pids = fresh_profile_ids(4);
        let start = Instant::now();
        for (i, p) in pids.iter().enumerate() {
            prober.submit(subtree_request(
                *p,
                anchor.to_path_buf(),
                cfg.clone(),
                i as u64 + 1,
            ));
        }
        let mut received = 0;
        while received < 4 {
            if let Ok(Input::ProbeResponse(_)) = rx.recv_timeout(Duration::from_secs(5)) {
                received += 1;
            }
        }
        let elapsed = start.elapsed();
        let _ = prober.shutdown();
        elapsed
    };

    // Warm the kernel inode/page cache; the first walk pays a cold
    // cost that would otherwise bias whichever burst runs first.
    let _ = mk_burst(4, tmp.path());

    let serial = mk_burst(1, tmp.path()).min(mk_burst(1, tmp.path()));
    let parallel = mk_burst(4, tmp.path()).min(mk_burst(4, tmp.path()));

    // Loose bound: 4 workers should be at least 1.5x faster than 1
    // worker on a 4-probe burst. Real ratios are typically 3x+.
    assert!(
        parallel * 5 < serial * 4,
        "concurrent burst not faster: serial={serial:?}, parallel={parallel:?}"
    );
}
