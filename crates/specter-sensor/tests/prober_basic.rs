//! `WorkerProber` round-trip smoke tests against real `tempfile::TempDir`
//! fixtures: File and Dir probes, Vanished on missing/kind-mismatch.

#![cfg(unix)]

use crossbeam::channel::unbounded;
use slotmap::SlotMap;
use specter_core::{
    ChildEntry, EntryKind, Input, ProbeCorrelation, ProbeKind, ProbeRequest, ProbeResult,
    ProfileId, ResourceId, ScanConfig, TreeSnapshot,
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

fn mk_request(
    profile: ProfileId,
    kind: ProbeKind,
    anchor: PathBuf,
    correlation: u64,
) -> ProbeRequest {
    ProbeRequest {
        profile,
        correlation: ProbeCorrelation(correlation),
        kind,
        target_resource: ResourceId::default(),
        target_path: anchor,
        scan_config: ScanConfig::builder().recursive(true).build(),
        captured_with: 0,
        baseline_subtree: None,
        force_walk: BTreeSet::new(),
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
fn probe_file_round_trip_emits_ok_with_one_entry() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("foo.c");
    std::fs::write(&path, b"hello").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&tx, 1).unwrap();

    let p = fresh_profile_id();
    prober.submit(mk_request(p, ProbeKind::File, path, 1));

    let resp = recv_response(&rx);
    assert_eq!(resp.profile, p);
    assert_eq!(resp.correlation, ProbeCorrelation(1));
    let ProbeResult::Ok(TreeSnapshot::File(leaf)) = resp.result else {
        panic!("expected Ok(File), got {:?}", resp.result);
    };
    assert_eq!(leaf.kind, EntryKind::File);
    assert_eq!(leaf.size, 5);

    let _ = prober.shutdown();
}

#[test]
fn probe_dir_round_trip_emits_ok_with_children() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/b.c"), b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&tx, 1).unwrap();

    let p = fresh_profile_id();
    prober.submit(mk_request(
        p,
        ProbeKind::Directory,
        tmp.path().to_path_buf(),
        7,
    ));

    let resp = recv_response(&rx);
    assert_eq!(resp.correlation, ProbeCorrelation(7));
    let ProbeResult::Ok(TreeSnapshot::Dir(arc)) = resp.result else {
        panic!("expected Ok(Dir)");
    };
    let names: Vec<&str> = arc
        .entries
        .keys()
        .map(compact_str::CompactString::as_str)
        .collect();
    assert!(names.contains(&"a.c"));
    assert!(names.contains(&"sub"));
    let sub_arc = match arc.entries.get("sub").expect("sub entry") {
        ChildEntry::Dir(dc) => dc.subtree.as_ref().expect("recursive subtree present"),
        ChildEntry::Leaf(_) => panic!("`sub` should be a Dir"),
    };
    let sub_names: Vec<&str> = sub_arc
        .entries
        .keys()
        .map(compact_str::CompactString::as_str)
        .collect();
    assert!(sub_names.contains(&"b.c"));

    let _ = prober.shutdown();
}

#[test]
fn probe_file_missing_yields_vanished() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("nope");

    let (tx, rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&tx, 1).unwrap();
    let p = fresh_profile_id();
    prober.submit(mk_request(p, ProbeKind::File, path, 1));

    let resp = recv_response(&rx);
    assert!(matches!(resp.result, ProbeResult::Vanished));

    let _ = prober.shutdown();
}

#[test]
fn probe_dir_missing_yields_vanished() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("nope");

    let (tx, rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&tx, 1).unwrap();
    let p = fresh_profile_id();
    prober.submit(mk_request(p, ProbeKind::Directory, path, 1));

    let resp = recv_response(&rx);
    assert!(matches!(resp.result, ProbeResult::Vanished));

    let _ = prober.shutdown();
}

#[test]
fn probe_file_on_directory_yields_vanished() {
    let tmp = TempDir::new().unwrap();

    let (tx, rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&tx, 1).unwrap();
    let p = fresh_profile_id();
    prober.submit(mk_request(p, ProbeKind::File, tmp.path().to_path_buf(), 1));

    let resp = recv_response(&rx);
    assert!(matches!(resp.result, ProbeResult::Vanished));

    let _ = prober.shutdown();
}

#[test]
fn probe_dir_on_file_yields_vanished() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&tx, 1).unwrap();
    let p = fresh_profile_id();
    prober.submit(mk_request(p, ProbeKind::Directory, path, 1));

    let resp = recv_response(&rx);
    assert!(matches!(resp.result, ProbeResult::Vanished));

    let _ = prober.shutdown();
}

#[test]
fn correlation_is_echoed_unchanged() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("f");
    std::fs::write(&path, b"x").unwrap();

    let (tx, rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&tx, 1).unwrap();
    let p = fresh_profile_id();
    prober.submit(mk_request(p, ProbeKind::File, path, 99));

    let resp = recv_response(&rx);
    assert_eq!(resp.correlation, ProbeCorrelation(99));
    assert_eq!(resp.profile, p);

    let _ = prober.shutdown();
}
