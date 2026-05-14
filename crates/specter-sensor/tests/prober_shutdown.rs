//! `shutdown(self)` cleanly drops the queue Sender, joins workers, and
//! returns each worker's `(index, thread::Result<()>)` for the bin to
//! log. `drop(prober)` without an explicit shutdown still terminates
//! workers via channel disconnect — workers run to completion on
//! pending probes, then exit.

#![cfg(unix)]

use crossbeam::channel::unbounded;
use slotmap::SlotMap;
use specter_core::{Input, ProbeCorrelation, ProbeOwner, ProbeRequest, ProfileId};
use specter_sensor::{Prober, WorkerProber};
use std::path::PathBuf;
use std::time::Duration;
use tempfile::TempDir;

fn fresh_profile_id() -> ProfileId {
    let mut sm = SlotMap::<ProfileId, ()>::with_key();
    sm.insert(())
}

fn mk_request(profile: ProfileId, target_path: PathBuf, correlation: u64) -> ProbeRequest {
    ProbeRequest::AnchorFile {
        owner: ProbeOwner::Profile(profile),
        correlation: ProbeCorrelation::from(correlation),
        target_path,
    }
}

#[test]
fn shutdown_with_no_pending_probes_returns_ok_per_worker() {
    let (tx, _rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&tx, 4).unwrap();
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
    let prober = WorkerProber::new(&tx, 2).unwrap();
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
        let prober = WorkerProber::new(&tx, 1).unwrap();
        let p = fresh_profile_id();
        prober.submit(mk_request(p, path, 1));
        // Wait for response before dropping — workers process the
        // pending probe before the queue Sender drop reaches them.
        let _ = rx.recv_timeout(Duration::from_secs(2)).expect("response");
        // Implicit drop here: queue_tx drops with prober → workers
        // see Disconnected on next recv → exit on their own. We don't
        // observe their JoinHandle (detached).
    }
    // After drop, any further sends to the engine_inbound channel
    // (via cloned Senders held by detached workers) won't happen —
    // workers have exited. The receiver should see no pending
    // messages.
    assert!(rx.try_recv().is_err(), "no spurious post-drop messages");
}
