//! Real-fs recursive walks with `max_depth`, `exclude`, `pattern`,
//! `hidden`. Pins the walker semantics through the
//! `WorkerProber` → `engine_inbound` channel surface.

#![cfg(unix)]

use crossbeam::channel::{Sender, unbounded};
use slotmap::SlotMap;
use specter_core::{
    ChildEntry, DirChild, DirSnapshot, GlobPattern, Input, ProbeCorrelation, ProbeOutcome,
    ProbeOwner, ProbeRequest, ProfileId, ProofObligation, ScanConfig,
};
use specter_sensor::{ProbeResponse, Prober, ProberResponseSender, SendError, WorkerProber};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use tempfile::TempDir;

fn fresh_profile_id() -> ProfileId {
    let mut sm = SlotMap::<ProfileId, ()>::with_key();
    sm.insert(())
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

fn segments(
    prober: &WorkerProber,
    rx: &crossbeam::channel::Receiver<Input>,
    anchor: PathBuf,
    cfg: ScanConfig,
) -> BTreeSet<String> {
    let p = fresh_profile_id();
    prober.submit(ProbeRequest::Subtree {
        owner: ProbeOwner::Profile(p),
        correlation: ProbeCorrelation::from(1),
        target_path: Arc::from(anchor),
        scan_config: cfg,
        captured_with: 0,
        baseline_subtree: None,
        obligation: ProofObligation::Chains(BTreeSet::new()),
        forced: false,
    });
    let resp = match rx.recv_timeout(Duration::from_secs(2)).expect("response") {
        Input::ProbeResponse(r) => r,
        other => panic!("unexpected: {other:?}"),
    };
    let ProbeOutcome::SubtreeProven { snapshot: arc, .. } = resp.outcome else {
        panic!("expected SubtreeProven");
    };
    let mut out = BTreeSet::new();
    collect_paths(&arc, "", &mut out);
    out
}

fn collect_paths(d: &DirSnapshot, prefix: &str, out: &mut BTreeSet<String>) {
    for (name, child) in d.entries() {
        let composed = if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}/{name}")
        };
        out.insert(composed.clone());
        if let ChildEntry::Dir(DirChild::Covered(sub)) = child {
            collect_paths(sub, &composed, out);
        }
    }
}

#[test]
fn recursive_walk_with_max_depth_three_collects_three_levels() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("a/b/c")).unwrap();
    std::fs::write(tmp.path().join("a/b/c/file.c"), b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), 1).unwrap();
    let cfg = ScanConfig::builder()
        .recursive(true)
        .max_depth(Some(3))
        .build();
    let segs = segments(&prober, &rx, tmp.path().to_path_buf(), cfg);
    // depth 1: a; 2: a/b; 3: a/b/c; 4 (file): excluded.
    assert!(segs.contains("a"));
    assert!(segs.contains("a/b"));
    assert!(segs.contains("a/b/c"));
    assert!(!segs.contains("a/b/c/file.c"));

    let _ = prober.shutdown();
}

#[test]
fn exclude_target_dir_omits_subtree_contents() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("target")).unwrap();
    std::fs::write(tmp.path().join("target/foo"), b"x").unwrap();
    std::fs::write(tmp.path().join("target/bar"), b"x").unwrap();
    std::fs::create_dir(tmp.path().join("src")).unwrap();
    std::fs::write(tmp.path().join("src/main.c"), b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), 1).unwrap();
    let cfg = ScanConfig::builder()
        .recursive(true)
        .exclude(GlobPattern::compile("target/**").unwrap())
        .build();
    let segs = segments(&prober, &rx, tmp.path().to_path_buf(), cfg);
    // No entry under `target/` is emitted (every cumulative path
    // starting with `target/` matches the glob).
    assert!(segs.iter().all(|s| !s.starts_with("target/")));
    assert!(segs.contains("src"));
    assert!(segs.contains("src/main.c"));

    let _ = prober.shutdown();
}

#[test]
fn pattern_double_star_matches_recursive_files() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("src")).unwrap();
    std::fs::write(tmp.path().join("src/main.c"), b"x").unwrap();
    std::fs::write(tmp.path().join("src/foo.txt"), b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), 1).unwrap();
    let cfg = ScanConfig::builder()
        .recursive(true)
        .pattern(GlobPattern::compile("**/*.c").unwrap())
        .build();
    let segs = segments(&prober, &rx, tmp.path().to_path_buf(), cfg);
    // Dirs always emit (pattern bypass); .c file emits; .txt
    // does not.
    assert!(segs.contains("src"));
    assert!(segs.contains("src/main.c"));
    assert!(!segs.contains("src/foo.txt"));

    let _ = prober.shutdown();
}

#[test]
fn hidden_false_skips_dot_subtree_entirely() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join(".git")).unwrap();
    std::fs::write(tmp.path().join(".git/HEAD"), b"x").unwrap();
    std::fs::write(tmp.path().join("main.c"), b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), 1).unwrap();
    let cfg = ScanConfig::builder().recursive(true).hidden(false).build();
    let segs = segments(&prober, &rx, tmp.path().to_path_buf(), cfg);
    assert!(!segs.contains(".git"));
    assert!(!segs.contains(".git/HEAD"));
    assert!(segs.contains("main.c"));

    let _ = prober.shutdown();
}

#[test]
fn hidden_true_includes_dot_subtree() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join(".git")).unwrap();
    std::fs::write(tmp.path().join(".git/HEAD"), b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let mut prober = WorkerProber::new(sink(tx), 1).unwrap();
    let cfg = ScanConfig::builder().recursive(true).hidden(true).build();
    let segs = segments(&prober, &rx, tmp.path().to_path_buf(), cfg);
    assert!(segs.contains(".git"));
    assert!(segs.contains(".git/HEAD"));

    let _ = prober.shutdown();
}
