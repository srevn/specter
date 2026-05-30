//! `WorkerProber` round-trip smoke tests against real `tempfile::TempDir`
//! fixtures: AnchorFile and Subtree probes, Vanished on missing/kind-mismatch.

#![cfg(unix)]

use crossbeam::channel::{Sender, unbounded};
use slotmap::SlotMap;
use specter_core::{
    EntryKind, Input, ProbeCorrelation, ProbeOutcome, ProbeOwner, ProbeRequest, ProfileId,
    ProofObligation, ScanConfig,
};
use specter_sensor::{ProbeResponse, Prober, ProberResponseSender, SendError, WorkerProber};
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
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

/// Mirror of the bin's `DriverProberSender` — wraps a single
/// `Sender<Input>` clone and rewraps each `ProbeResponse` as
/// `Input::ProbeResponse(_)` on the wire.
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
        owner: ProbeOwner::Profile(profile),
        correlation: ProbeCorrelation::from(correlation),
        target_path: Arc::from(target_path),
    }
}

fn subtree_request(profile: ProfileId, target_path: PathBuf, correlation: u64) -> ProbeRequest {
    let target_path: Arc<Path> = Arc::from(target_path);
    let anchor_path = Arc::clone(&target_path);
    ProbeRequest::Subtree {
        owner: ProbeOwner::Profile(profile),
        correlation: ProbeCorrelation::from(correlation),
        target_path,
        anchor_path,
        scan_config: ScanConfig::builder().recursive(true).build(),
        captured_with: 0,
        baseline_subtree: None,
        obligation: ProofObligation::WholeSubtree,
        forced: false,
    }
}

fn recv_response(rx: &crossbeam::channel::Receiver<Input>) -> specter_core::ProbeResponse {
    match rx
        .recv_timeout(Duration::from_secs(2))
        .expect("response within timeout")
    {
        Input::ProbeResponse(r) => r,
        other => panic!("unexpected input: {other:?}"),
    }
}

#[test]
fn anchor_file_round_trip_emits_anchor_ok_with_leaf() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("foo.c");
    std::fs::write(&path, b"hello").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), nz(1)).unwrap();

    let p = fresh_profile_id();
    prober.submit(anchor_request(p, path, 1));

    let resp = recv_response(&rx);
    assert_eq!(resp.owner, ProbeOwner::Profile(p));
    assert_eq!(resp.correlation, ProbeCorrelation::from(1));
    let ProbeOutcome::AnchorOk(leaf) = resp.outcome else {
        panic!("expected AnchorOk, got {:?}", resp.outcome);
    };
    assert_eq!(leaf.kind(), EntryKind::File);
    assert_eq!(leaf.size(), 5);

    let _ = prober.shutdown();
}

#[test]
fn subtree_round_trip_emits_subtree_ok_with_children() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/b.c"), b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), nz(1)).unwrap();

    let p = fresh_profile_id();
    prober.submit(subtree_request(p, tmp.path().to_path_buf(), 7));

    let resp = recv_response(&rx);
    assert_eq!(resp.correlation, ProbeCorrelation::from(7));
    let ProbeOutcome::SubtreeProven { snapshot: arc, .. } = resp.outcome else {
        panic!("expected SubtreeProven");
    };
    let names: Vec<&str> = arc
        .entries()
        .keys()
        .map(compact_str::CompactString::as_str)
        .collect();
    assert!(names.contains(&"a.c"));
    assert!(names.contains(&"sub"));
    let sub_arc = arc
        .lookup_covered_dir("sub")
        .expect("recursive subtree present");
    assert!(sub_arc.entries().contains_key("b.c"));

    let _ = prober.shutdown();
}

#[test]
fn anchor_file_missing_yields_vanished() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("nope");

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), nz(1)).unwrap();
    let p = fresh_profile_id();
    prober.submit(anchor_request(p, path, 1));

    let resp = recv_response(&rx);
    assert!(matches!(resp.outcome, ProbeOutcome::Vanished));

    let _ = prober.shutdown();
}

#[test]
fn subtree_missing_yields_vanished() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("nope");

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), nz(1)).unwrap();
    let p = fresh_profile_id();
    prober.submit(subtree_request(p, path, 1));

    let resp = recv_response(&rx);
    assert!(matches!(resp.outcome, ProbeOutcome::Vanished));

    let _ = prober.shutdown();
}

#[test]
fn anchor_file_on_directory_yields_vanished() {
    let tmp = TempDir::new().unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), nz(1)).unwrap();
    let p = fresh_profile_id();
    prober.submit(anchor_request(p, tmp.path().to_path_buf(), 1));

    let resp = recv_response(&rx);
    assert!(matches!(resp.outcome, ProbeOutcome::Vanished));

    let _ = prober.shutdown();
}

#[test]
fn subtree_on_file_yields_vanished() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), nz(1)).unwrap();
    let p = fresh_profile_id();
    prober.submit(subtree_request(p, path, 1));

    let resp = recv_response(&rx);
    assert!(matches!(resp.outcome, ProbeOutcome::Vanished));

    let _ = prober.shutdown();
}

#[test]
fn correlation_is_echoed_unchanged() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("f");
    std::fs::write(&path, b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), nz(1)).unwrap();
    let p = fresh_profile_id();
    prober.submit(anchor_request(p, path, 99));

    let resp = recv_response(&rx);
    assert_eq!(resp.correlation, ProbeCorrelation::from(99));
    assert_eq!(resp.owner, ProbeOwner::Profile(p));

    let _ = prober.shutdown();
}
