//! A walk that runs out of descriptors says nothing of the tree. With none free, the walk root's
//! open fails `EMFILE` and the probe answers `Failed(Transient)`; the walk holds one directory open
//! at a time, and every open goes through the same classifier, so the root's open stands for all of
//! them. Its own binary: the lowered `RLIMIT_NOFILE` is process-wide.

#![cfg(unix)]

use crossbeam::channel::unbounded;
use nix::sys::resource::{Resource, getrlimit, setrlimit};
use slotmap::SlotMap;
use specter_core::{
    Input, ProbeCorrelation, ProbeFailure, ProbeOutcome, ProbeRequest, ProfileId, ProofObligation,
    ScanConfig,
};
use specter_sensor::{ProbeResponse, Prober, ProberResponseSender, SendError, WorkerProber};
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

struct Sink(crossbeam::channel::Sender<Input>);

impl ProberResponseSender for Sink {
    fn send(&self, response: ProbeResponse) -> Result<(), SendError> {
        self.0
            .send(Input::ProbeResponse(response))
            .map_err(|_| SendError::Disconnected)
    }
}

#[test]
fn a_walk_with_no_descriptor_free_fails_transient() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("f"), b"x").unwrap();
    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(
        Arc::new(Sink(tx)),
        NonZeroUsize::new(1).expect("one worker"),
    )
    .unwrap();
    let owner: ProfileId = SlotMap::with_key().insert(());
    let root: Arc<Path> = Arc::from(tmp.path());

    // Lower the soft limit and take every descriptor it leaves.
    let (_, hard) = getrlimit(Resource::RLIMIT_NOFILE).expect("getrlimit");
    setrlimit(Resource::RLIMIT_NOFILE, 64, hard).expect("setrlimit");
    let mut held = Vec::new();
    while let Ok(file) = std::fs::File::open("/dev/null") {
        held.push(file);
    }

    prober.submit(ProbeRequest::Subtree {
        owner,
        correlation: ProbeCorrelation::from(1),
        target_path: Arc::clone(&root),
        anchor_path: root,
        scan_config: Arc::new(ScanConfig::builder().recursive(true).build()),
        captured_with: 0,
        baseline_subtree: None,
        obligation: ProofObligation::WholeSubtree,
        forced: false,
    });
    let response = rx.recv_timeout(Duration::from_secs(10));
    // The descriptors go back before `TempDir` needs one to remove the tree.
    drop(held);
    let _ = prober.shutdown();

    let Ok(Input::ProbeResponse(response)) = response else {
        panic!("expected a probe response, got {response:?}");
    };
    assert!(
        matches!(
            response.outcome,
            ProbeOutcome::Failed(ProbeFailure::Transient {
                errno: libc::EMFILE
            })
        ),
        "got {:?}",
        response.outcome,
    );
}
