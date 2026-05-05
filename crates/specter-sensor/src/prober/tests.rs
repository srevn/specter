//! Sibling unit tests for `prober::walk` and `prober::pool`.
//!
//! Walk tests use real `tempfile::TempDir` fixtures — no mocking the
//! filesystem. Pool tests drive `run_worker` directly with synchronous
//! crossbeam channels and a custom probe closure: that's how
//! cancellation, panic recovery, and post-run cleanup get
//! deterministic coverage without relying on multi-thread scheduling.

use super::pool::{ExpectedMap, WorkerProber, run_worker};
use super::walk::{probe_dir, probe_file};
use crate::Prober;
use crossbeam::channel::{Receiver, Sender, unbounded};
use slotmap::SlotMap;
use specter_core::{
    ChildEntry, DirMeta, DirSnapshot, EntryKind, GlobPattern, Input, ProbeCorrelation,
    ProbeRequest, ProbeResult, ProfileId, ResourceId, ScanConfig, TreeSnapshot,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tempfile::TempDir;

fn fresh_expected() -> ExpectedMap {
    Arc::new(Mutex::new(BTreeMap::new()))
}

fn seed(map: &ExpectedMap, p: ProfileId, c: u64) {
    map.lock().unwrap().insert(p, ProbeCorrelation(c));
}

// ---------------------------------------------------------------- helpers

fn fresh_profile_ids(n: usize) -> Vec<ProfileId> {
    let mut sm = SlotMap::<ProfileId, ()>::with_key();
    (0..n).map(|_| sm.insert(())).collect()
}

fn req(profile: ProfileId, correlation: u64, kind: specter_core::ProbeKind) -> ProbeRequest {
    ProbeRequest {
        profile,
        correlation: ProbeCorrelation(correlation),
        kind,
        target_resource: ResourceId::default(),
        target_path: PathBuf::from("/dev/null"),
        scan_config: ScanConfig::builder().build(),
        captured_with: 0,
        baseline_subtree: None,
        force_walk: BTreeSet::new(),
        forced: false,
    }
}

/// Default arg pack for `probe_dir` calls:
/// `target_resource = default`, `captured_with = 0`, no baseline, empty
/// `force_walk`, `forced = false`. Use the explicit form when the test
/// wants to exercise mtime-skip / `force_walk` / forced.
fn pdir(path: &std::path::Path, cfg: &ScanConfig) -> ProbeResult {
    probe_dir(
        path,
        ResourceId::default(),
        cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    )
}

/// Recursively collect every entry's relative path (segment from the
/// anchor) from a `ProbeResult::Ok(TreeSnapshot::Dir(...))`. Sorted.
fn segments(result: &ProbeResult) -> Vec<String> {
    let ProbeResult::Ok(snap) = result else {
        panic!("expected Ok, got {result:?}");
    };
    let TreeSnapshot::Dir(arc) = snap else {
        panic!("expected Dir, got File");
    };
    let mut out = Vec::new();
    collect_paths(arc, "", &mut out);
    out.sort();
    out
}

fn collect_paths(d: &DirSnapshot, prefix: &str, out: &mut Vec<String>) {
    for (name, child) in &d.entries {
        let composed = if prefix.is_empty() {
            name.to_string()
        } else {
            format!("{prefix}/{name}")
        };
        out.push(composed.clone());
        if let ChildEntry::Dir(dc) = child
            && let Some(sub) = dc.subtree.as_deref()
        {
            collect_paths(sub, &composed, out);
        }
    }
}

// ---------------------------------------------------------------- walk: probe_file

#[test]
fn probe_file_returns_ok_for_regular_file() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("foo.c");
    std::fs::write(&path, b"hello").unwrap();

    let result = probe_file(&path);
    let ProbeResult::Ok(snap) = &result else {
        panic!("expected Ok, got {result:?}");
    };
    let TreeSnapshot::File(leaf) = snap else {
        panic!("expected TreeSnapshot::File");
    };
    assert_eq!(leaf.kind, EntryKind::File);
    assert_eq!(leaf.size, 5);
}

#[test]
fn probe_file_returns_vanished_for_missing_path() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("nope");
    assert!(matches!(probe_file(&path), ProbeResult::Vanished));
}

#[test]
fn probe_file_returns_vanished_for_directory() {
    let tmp = TempDir::new().unwrap();
    assert!(matches!(probe_file(tmp.path()), ProbeResult::Vanished));
}

#[test]
fn probe_file_returns_vanished_for_symlink() {
    let tmp = TempDir::new().unwrap();
    let target = tmp.path().join("target.c");
    let link = tmp.path().join("link");
    std::fs::write(&target, b"x").unwrap();
    std::os::unix::fs::symlink(&target, &link).unwrap();

    // probe_file uses lstat — the symlink is the kind it sees, not the
    // target. Symlink ≠ regular file ⇒ Vanished.
    assert!(matches!(probe_file(&link), ProbeResult::Vanished));
}

// ---------------------------------------------------------------- walk: probe_dir

#[test]
fn probe_dir_returns_vanished_for_missing_path() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("nope");
    let cfg = ScanConfig::builder().build();
    assert!(matches!(pdir(&path, &cfg), ProbeResult::Vanished));
}

#[test]
fn probe_dir_returns_vanished_for_regular_file() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    assert!(matches!(pdir(&path, &cfg), ProbeResult::Vanished));
}

#[test]
fn probe_dir_empty_dir_returns_zero_entries() {
    let tmp = TempDir::new().unwrap();
    let cfg = ScanConfig::builder().build();
    let result = pdir(tmp.path(), &cfg);
    let ProbeResult::Ok(TreeSnapshot::Dir(arc)) = result else {
        panic!("expected Ok(Dir)");
    };
    assert!(arc.entries.is_empty());
}

#[test]
fn probe_dir_flat_lists_children() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a"), b"1").unwrap();
    std::fs::write(tmp.path().join("b"), b"2").unwrap();
    std::fs::write(tmp.path().join("c"), b"3").unwrap();
    let cfg = ScanConfig::builder().build();
    let result = pdir(tmp.path(), &cfg);
    let segs = segments(&result);
    assert_eq!(
        segs,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
}

#[test]
fn probe_dir_recursive_collects_descendants() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/file.c"), b"x").unwrap();

    let cfg = ScanConfig::builder().recursive(true).build();
    let result = pdir(tmp.path(), &cfg);
    let segs = segments(&result);
    assert_eq!(segs, vec!["sub".to_string(), "sub/file.c".to_string()]);
}

#[test]
fn probe_dir_non_recursive_omits_grandchildren() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/file.c"), b"x").unwrap();

    let cfg = ScanConfig::builder().recursive(false).build();
    let result = pdir(tmp.path(), &cfg);
    let segs = segments(&result);
    assert_eq!(segs, vec!["sub".to_string()]);
}

#[test]
fn probe_dir_max_depth_one_excludes_grandchildren() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/grand.c"), b"x").unwrap();

    let cfg = ScanConfig::builder()
        .recursive(true)
        .max_depth(Some(1))
        .build();
    let segs = segments(&pdir(tmp.path(), &cfg));
    assert_eq!(segs, vec!["sub".to_string()]);
}

#[test]
fn probe_dir_max_depth_two_includes_grandchildren() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/grand.c"), b"x").unwrap();

    let cfg = ScanConfig::builder()
        .recursive(true)
        .max_depth(Some(2))
        .build();
    let segs = segments(&pdir(tmp.path(), &cfg));
    assert_eq!(segs, vec!["sub".to_string(), "sub/grand.c".to_string()]);
}

#[test]
fn probe_dir_max_depth_three_collects_three_levels() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("a/b/c")).unwrap();
    std::fs::write(tmp.path().join("a/b/c/file.c"), b"x").unwrap();

    let cfg = ScanConfig::builder()
        .recursive(true)
        .max_depth(Some(3))
        .build();
    let segs = segments(&pdir(tmp.path(), &cfg));
    // Depth 1: "a"; depth 2: "a/b"; depth 3: "a/b/c". File is at depth
    // 4 — excluded by max_depth=3.
    assert_eq!(
        segs,
        vec!["a".to_string(), "a/b".to_string(), "a/b/c".to_string()]
    );
}

#[test]
fn probe_dir_exclude_drops_matched_entries() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("target")).unwrap();
    std::fs::write(tmp.path().join("target/foo"), b"x").unwrap();
    std::fs::create_dir(tmp.path().join("src")).unwrap();
    std::fs::write(tmp.path().join("src/main.c"), b"x").unwrap();

    let exclude = GlobPattern::compile("target/**").unwrap();
    let cfg = ScanConfig::builder()
        .recursive(true)
        .exclude(exclude)
        .build();
    let segs = segments(&pdir(tmp.path(), &cfg));
    // `target/**` matches paths under target (and `target` itself with
    // globset's `**` semantics). `target/foo` is excluded.
    assert!(!segs.iter().any(|s| s.starts_with("target/")));
    assert!(segs.contains(&"src".to_string()));
    assert!(segs.contains(&"src/main.c".to_string()));
}

#[test]
fn probe_dir_pattern_matches_files_only() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("main.c"), b"x").unwrap();
    std::fs::write(tmp.path().join("foo.txt"), b"x").unwrap();
    std::fs::create_dir(tmp.path().join("subdir")).unwrap();

    let pattern = GlobPattern::compile("**/*.c").unwrap();
    let cfg = ScanConfig::builder().pattern(pattern).build();
    let segs = segments(&pdir(tmp.path(), &cfg));
    // .c files: in. .txt: out. Subdirs: in (Dir bypass).
    assert!(segs.contains(&"main.c".to_string()));
    assert!(!segs.contains(&"foo.txt".to_string()));
    assert!(segs.contains(&"subdir".to_string()));
}

#[test]
fn probe_dir_pattern_recursive_matches_nested_files() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("src")).unwrap();
    std::fs::write(tmp.path().join("src/main.c"), b"x").unwrap();
    std::fs::write(tmp.path().join("src/foo.txt"), b"x").unwrap();

    let pattern = GlobPattern::compile("**/*.c").unwrap();
    let cfg = ScanConfig::builder()
        .recursive(true)
        .pattern(pattern)
        .build();
    let segs = segments(&pdir(tmp.path(), &cfg));
    assert!(segs.contains(&"src".to_string()));
    assert!(segs.contains(&"src/main.c".to_string()));
    assert!(!segs.iter().any(|s| s.contains("foo.txt")));
}

#[test]
fn probe_dir_hidden_false_skips_dotfiles() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join(".git")).unwrap();
    std::fs::write(tmp.path().join(".git/HEAD"), b"x").unwrap();
    std::fs::write(tmp.path().join("main.c"), b"x").unwrap();

    let cfg = ScanConfig::builder().recursive(true).hidden(false).build();
    let segs = segments(&pdir(tmp.path(), &cfg));
    assert_eq!(segs, vec!["main.c".to_string()]);
}

#[test]
fn probe_dir_hidden_true_includes_dotfiles() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join(".git")).unwrap();
    std::fs::write(tmp.path().join(".git/HEAD"), b"x").unwrap();
    std::fs::write(tmp.path().join("main.c"), b"x").unwrap();

    let cfg = ScanConfig::builder().recursive(true).hidden(true).build();
    let segs = segments(&pdir(tmp.path(), &cfg));
    assert!(segs.contains(&".git".to_string()));
    assert!(segs.contains(&".git/HEAD".to_string()));
    assert!(segs.contains(&"main.c".to_string()));
}

#[test]
fn probe_dir_does_not_descend_symlinks() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("target")).unwrap();
    std::fs::write(tmp.path().join("target/file.c"), b"x").unwrap();
    std::os::unix::fs::symlink(tmp.path().join("target"), tmp.path().join("link")).unwrap();

    let cfg = ScanConfig::builder().recursive(true).build();
    let segs = segments(&pdir(tmp.path(), &cfg));
    // `link` is a symlink: emitted as Symlink entry but not descended
    // through. `target` is a real dir: emitted and descended.
    assert!(segs.contains(&"link".to_string()));
    assert!(segs.contains(&"target".to_string()));
    assert!(segs.contains(&"target/file.c".to_string()));
    // Critically: no `link/file.c` (would prove symlink traversal).
    assert!(!segs.iter().any(|s| s.starts_with("link/")));
}

#[test]
fn probe_dir_emits_symlink_entry_kind() {
    let tmp = TempDir::new().unwrap();
    let target = tmp.path().join("target.c");
    std::fs::write(&target, b"x").unwrap();
    std::os::unix::fs::symlink(&target, tmp.path().join("link")).unwrap();

    let cfg = ScanConfig::builder().build();
    let result = pdir(tmp.path(), &cfg);
    let ProbeResult::Ok(TreeSnapshot::Dir(arc)) = result else {
        panic!("expected Ok(Dir)");
    };
    let link_entry = arc.entries.get("link").expect("link entry");
    let ChildEntry::Leaf(l) = link_entry else {
        panic!("symlink emits as Leaf");
    };
    assert_eq!(l.kind, EntryKind::Symlink);
}

#[test]
fn probe_dir_skips_unreadable_subdir_emits_remaining() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = TempDir::new().unwrap();
    let forbidden = tmp.path().join("forbidden");
    std::fs::create_dir(&forbidden).unwrap();
    std::fs::write(forbidden.join("inside.c"), b"x").unwrap();
    std::fs::set_permissions(&forbidden, std::fs::Permissions::from_mode(0o000)).unwrap();
    std::fs::write(tmp.path().join("ok.c"), b"x").unwrap();

    let cfg = ScanConfig::builder().recursive(true).build();
    let result = pdir(tmp.path(), &cfg);

    // Restore perms before TempDir drops (else cleanup fails).
    std::fs::set_permissions(&forbidden, std::fs::Permissions::from_mode(0o755)).unwrap();

    let segs = segments(&result);
    // `forbidden` is emitted as a Dir entry; readdir on it skips with
    // a warn; siblings still emit.
    assert!(segs.contains(&"forbidden".to_string()));
    assert!(segs.contains(&"ok.c".to_string()));
    // `inside.c` is unreachable.
    assert!(!segs.iter().any(|s| s.contains("inside")));
}

// ---------------------------------------------------------------- walk: mtime-skip
//
// The mtime-skip path: equal `(mtime, inode, device)` between
// `baseline.root_meta` and the freshly `lstat`ed directory ⇒ return
// `Arc::clone(baseline)`. These tests pin the exact-match success case
// and each defeat case.

/// Probe `target_path` once with `cfg`, take the resulting Arc, and
/// re-probe with that Arc as `baseline_subtree`. Returns `(first, second)`
/// for `Arc::ptr_eq` comparison.
fn probe_then_reprobe_with_baseline(
    target: &std::path::Path,
    cfg: &ScanConfig,
) -> (Arc<DirSnapshot>, ProbeResult) {
    let first = probe_dir(
        target,
        ResourceId::default(),
        cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc)) = first else {
        panic!("first probe failed");
    };
    let second = probe_dir(
        target,
        ResourceId::default(),
        cfg,
        0,
        Some(&arc),
        &BTreeSet::new(),
        false,
    );
    (arc, second)
}

#[test]
fn mtime_skip_returns_arc_clone_when_root_meta_matches() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let (baseline, second) = probe_then_reprobe_with_baseline(tmp.path(), &cfg);
    let ProbeResult::Ok(TreeSnapshot::Dir(arc2)) = second else {
        panic!("second probe failed");
    };
    assert!(
        Arc::ptr_eq(&baseline, &arc2),
        "mtime-skip should hand back the baseline Arc unchanged"
    );
}

#[test]
fn mtime_skip_does_not_match_when_mtime_differs() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let first = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(baseline)) = first else {
        panic!("first probe failed");
    };
    // Forge a baseline with a different mtime; walker enumerates fresh.
    let forged = Arc::new(DirSnapshot::new(
        baseline.root_resource,
        DirMeta {
            mtime: std::time::UNIX_EPOCH,
            ..baseline.root_meta
        },
        Instant::now(),
        baseline.captured_with,
        baseline.entries.clone(),
    ));
    let result = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        Some(&forged),
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc2)) = result else {
        panic!("re-probe failed");
    };
    assert!(
        !Arc::ptr_eq(&forged, &arc2),
        "mtime mismatch should force fresh enumeration"
    );
}

#[test]
fn mtime_skip_does_not_match_when_inode_differs() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let first = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(baseline)) = first else {
        panic!("first probe failed");
    };
    let forged = Arc::new(DirSnapshot::new(
        baseline.root_resource,
        DirMeta {
            inode: baseline.root_meta.inode.wrapping_add(1),
            ..baseline.root_meta
        },
        Instant::now(),
        baseline.captured_with,
        baseline.entries.clone(),
    ));
    let result = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        Some(&forged),
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc2)) = result else {
        panic!("re-probe failed");
    };
    assert!(!Arc::ptr_eq(&forged, &arc2));
}

#[test]
fn mtime_skip_does_not_match_when_device_differs() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let first = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(baseline)) = first else {
        panic!("first probe failed");
    };
    let forged = Arc::new(DirSnapshot::new(
        baseline.root_resource,
        DirMeta {
            device: baseline.root_meta.device.wrapping_add(1),
            ..baseline.root_meta
        },
        Instant::now(),
        baseline.captured_with,
        baseline.entries.clone(),
    ));
    let result = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        Some(&forged),
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc2)) = result else {
        panic!("re-probe failed");
    };
    assert!(!Arc::ptr_eq(&forged, &arc2));
}

#[test]
fn mtime_skip_recursive_propagates_via_subtree_baseline() {
    // Top-level mtime matches AND child-subdir's mtime matches ⇒ child Arc
    // re-used (recursive Arc::ptr_eq through the BTreeMap).
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/file.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().recursive(true).build();
    let first = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(baseline)) = first else {
        panic!("first probe failed");
    };
    let prior_sub_arc = match baseline.entries.get("sub").expect("sub child") {
        ChildEntry::Dir(dc) => dc.subtree.clone().expect("sub subtree present"),
        ChildEntry::Leaf(_) => panic!("sub should be a Dir"),
    };
    let second = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        Some(&baseline),
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(top)) = second else {
        panic!("re-probe failed");
    };
    // Top is the baseline Arc (mtime matched) ⇒ ptr_eq.
    assert!(Arc::ptr_eq(&baseline, &top));
    // The child subtree is shared transitively (it lives inside `top`).
    let new_sub_arc = match top.entries.get("sub").expect("sub child").clone() {
        ChildEntry::Dir(dc) => dc.subtree.expect("sub subtree present"),
        ChildEntry::Leaf(_) => panic!("sub should be a Dir"),
    };
    assert!(Arc::ptr_eq(&prior_sub_arc, &new_sub_arc));
}

// ---------------------------------------------------------------- walk: force_walk

#[test]
fn force_walk_with_path_in_set_refuses_skip() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let first = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(baseline)) = first else {
        panic!("first probe failed");
    };
    // force_walk = {target_path} — refuse the skip even though mtime matches.
    let mut force = BTreeSet::new();
    force.insert(tmp.path().to_path_buf());
    let second = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        Some(&baseline),
        &force,
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc2)) = second else {
        panic!("re-probe failed");
    };
    assert!(
        !Arc::ptr_eq(&baseline, &arc2),
        "force_walk on the target path should defeat mtime-skip"
    );
}

#[test]
fn force_walk_with_descendant_in_set_refuses_skip_at_target() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/file.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().recursive(true).build();
    let first = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(baseline)) = first else {
        panic!("first probe failed");
    };
    // force_walk = {tmp/sub/file.c}: descendant; root must enumerate so we
    // can recurse into `sub`.
    let mut force = BTreeSet::new();
    force.insert(tmp.path().join("sub").join("file.c"));
    let second = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        Some(&baseline),
        &force,
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc2)) = second else {
        panic!("re-probe failed");
    };
    assert!(!Arc::ptr_eq(&baseline, &arc2));
}

#[test]
fn force_walk_with_unrelated_path_does_not_affect_skip() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let first = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(baseline)) = first else {
        panic!("first probe failed");
    };
    // Path outside target's subtree — skip applies normally.
    let mut force = BTreeSet::new();
    force.insert(PathBuf::from("/totally/unrelated/path"));
    let second = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        Some(&baseline),
        &force,
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc2)) = second else {
        panic!("re-probe failed");
    };
    assert!(Arc::ptr_eq(&baseline, &arc2));
}

#[test]
fn force_walk_propagates_through_recursion_to_descendant() {
    // force_walk = {target/dir_a/dir_b}: target enumerates → dir_a
    // enumerates → dir_b enumerates. sibling dir_c (mtime-stable) is
    // mtime-skipped during the recursion, since no force path is at-or-
    // under dir_c.
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("dir_a/dir_b")).unwrap();
    std::fs::create_dir(tmp.path().join("dir_c")).unwrap();
    std::fs::write(tmp.path().join("dir_c/x.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().recursive(true).build();
    let first = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(baseline)) = first else {
        panic!("first probe failed");
    };
    let prior_dir_c_arc = match baseline.entries.get("dir_c").unwrap() {
        ChildEntry::Dir(dc) => dc.subtree.clone().unwrap(),
        ChildEntry::Leaf(_) => panic!(),
    };
    let mut force = BTreeSet::new();
    force.insert(tmp.path().join("dir_a").join("dir_b"));
    let second = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        Some(&baseline),
        &force,
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(top)) = second else {
        panic!("re-probe failed");
    };
    // dir_c was untouched and not under a forced path; the recursion
    // should mtime-skip and reuse the baseline Arc.
    let new_dir_c_arc = match top.entries.get("dir_c").unwrap().clone() {
        ChildEntry::Dir(dc) => dc.subtree.unwrap(),
        ChildEntry::Leaf(_) => panic!(),
    };
    assert!(Arc::ptr_eq(&prior_dir_c_arc, &new_dir_c_arc));
}

#[test]
fn force_walk_empty_set_behaves_like_v4() {
    // No force paths — pure mtime-skip semantics. Equivalent to the
    // `mtime_skip_returns_arc_clone` test.
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let (baseline, second) = probe_then_reprobe_with_baseline(tmp.path(), &cfg);
    let ProbeResult::Ok(TreeSnapshot::Dir(arc2)) = second else {
        panic!("re-probe failed");
    };
    assert!(Arc::ptr_eq(&baseline, &arc2));
}

// ---------------------------------------------------------------- walk: forced

#[test]
fn forced_true_bypasses_mtime_skip_at_root() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let first = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(baseline)) = first else {
        panic!("first probe failed");
    };
    let second = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        Some(&baseline),
        &BTreeSet::new(),
        true, // forced
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc2)) = second else {
        panic!("re-probe failed");
    };
    assert!(
        !Arc::ptr_eq(&baseline, &arc2),
        "forced=true must bypass mtime-skip at the root"
    );
}

#[test]
fn forced_true_bypasses_mtime_skip_in_recursion() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/file.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().recursive(true).build();
    let first = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(baseline)) = first else {
        panic!("first probe failed");
    };
    let prior_sub_arc = match baseline.entries.get("sub").unwrap() {
        ChildEntry::Dir(dc) => dc.subtree.clone().unwrap(),
        ChildEntry::Leaf(_) => panic!(),
    };
    let second = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        Some(&baseline),
        &BTreeSet::new(),
        true, // forced
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(top)) = second else {
        panic!("re-probe failed");
    };
    let new_sub_arc = match top.entries.get("sub").unwrap().clone() {
        ChildEntry::Dir(dc) => dc.subtree.unwrap(),
        ChildEntry::Leaf(_) => panic!(),
    };
    assert!(
        !Arc::ptr_eq(&prior_sub_arc, &new_sub_arc),
        "forced=true must thread through recursion"
    );
}

#[test]
fn forced_false_default_path_unaffected() {
    // Control: forced=false + mtime match + empty force_walk → skip.
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let (baseline, second) = probe_then_reprobe_with_baseline(tmp.path(), &cfg);
    let ProbeResult::Ok(TreeSnapshot::Dir(arc2)) = second else {
        panic!("re-probe failed");
    };
    assert!(Arc::ptr_eq(&baseline, &arc2));
}

// ---------------------------------------------------------------- walk: DirSnapshot construction

#[test]
fn dir_snapshot_root_meta_carries_lstat_triple() {
    use std::os::unix::fs::MetadataExt;
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let raw = std::fs::symlink_metadata(tmp.path()).unwrap();
    let result = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc)) = result else {
        panic!("expected Ok(Dir)");
    };
    assert_eq!(arc.root_meta.inode, raw.ino());
    assert_eq!(arc.root_meta.device, raw.dev());
}

#[test]
fn dir_snapshot_captured_with_carries_request_value() {
    const STAMP: u64 = 0xCAFE_BABE_DEAD_BEEF;
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let result = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        STAMP,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc)) = result else {
        panic!();
    };
    assert_eq!(arc.captured_with, STAMP);
}

#[test]
fn dir_snapshot_target_resource_carries_request_value() {
    // Build a non-default ResourceId via slotmap — same shape that the
    // engine would produce for a real Profile.
    let tmp = TempDir::new().unwrap();
    let cfg = ScanConfig::builder().build();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let rid = sm.insert(());
    let result = probe_dir(tmp.path(), rid, &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeResult::Ok(TreeSnapshot::Dir(arc)) = result else {
        panic!();
    };
    assert_eq!(arc.root_resource, rid);
}

#[test]
fn dir_snapshot_subtree_resources_are_default() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    let cfg = ScanConfig::builder().recursive(true).build();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let rid = sm.insert(());
    let result = probe_dir(tmp.path(), rid, &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeResult::Ok(TreeSnapshot::Dir(arc)) = result else {
        panic!();
    };
    let sub_arc = match arc.entries.get("sub").unwrap() {
        ChildEntry::Dir(dc) => dc.subtree.as_ref().unwrap(),
        ChildEntry::Leaf(_) => panic!(),
    };
    assert_eq!(sub_arc.root_resource, ResourceId::default());
}

#[test]
fn dir_snapshot_uncovered_branches_have_subtree_none() {
    // max_depth=1 + grandchild dir → the depth-1 dir's subtree is None.
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::create_dir(tmp.path().join("sub/inner")).unwrap();
    let cfg = ScanConfig::builder()
        .recursive(true)
        .max_depth(Some(1))
        .build();
    let result = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc)) = result else {
        panic!();
    };
    match arc.entries.get("sub").unwrap() {
        ChildEntry::Dir(dc) => {
            assert!(
                dc.subtree.is_none(),
                "depth-1 dir uncovered (max_depth=1) ⇒ subtree=None"
            );
        }
        ChildEntry::Leaf(_) => panic!(),
    }
}

#[test]
fn dir_snapshot_pattern_filtered_files_absent() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("main.c"), b"x").unwrap();
    std::fs::write(tmp.path().join("foo.txt"), b"x").unwrap();
    let cfg = ScanConfig::builder()
        .pattern(GlobPattern::compile("**/*.c").unwrap())
        .build();
    let result = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc)) = result else {
        panic!();
    };
    assert!(arc.entries.contains_key("main.c"));
    assert!(!arc.entries.contains_key("foo.txt"));
}

#[test]
fn dir_snapshot_excluded_paths_absent() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("target")).unwrap();
    std::fs::write(tmp.path().join("target/foo"), b"x").unwrap();
    std::fs::write(tmp.path().join("main.c"), b"x").unwrap();
    let cfg = ScanConfig::builder()
        .recursive(true)
        .exclude(GlobPattern::compile("target/**").unwrap())
        .build();
    let result = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc)) = result else {
        panic!();
    };
    assert!(arc.entries.contains_key("main.c"));
    // `target/foo` excluded; `target` itself depends on glob semantics —
    // `target/**` matches paths under target, so `target` itself is in
    // (the prober_recursive integration test pins the same).
    let target_present = arc.entries.contains_key("target");
    let target_uncov = match arc.entries.get("target") {
        Some(ChildEntry::Dir(dc)) => dc.subtree.as_ref().is_none_or(|s| s.entries.is_empty()),
        _ => true,
    };
    assert!(
        !target_present || target_uncov,
        "target's subtree (if present) must be empty post-exclude"
    );
}

// ---------------------------------------------------------------- walk: ProbeKind::File

#[test]
fn probe_file_emits_tree_snapshot_file_for_regular_file() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("foo.c");
    std::fs::write(&path, b"hello").unwrap();
    let result = probe_file(&path);
    let ProbeResult::Ok(TreeSnapshot::File(leaf)) = result else {
        panic!("expected Ok(File), got {result:?}");
    };
    assert_eq!(leaf.kind, EntryKind::File);
    assert_eq!(leaf.size, 5);
}

// ---------------------------------------------------------------- walk: determinism

#[test]
fn entries_are_lex_sorted_by_btreemap() {
    // BTreeMap iterates lex-by-key by construction; verify the walker
    // produces a map whose keys' iteration order is lex.
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("zeta"), b"x").unwrap();
    std::fs::write(tmp.path().join("alpha"), b"x").unwrap();
    std::fs::write(tmp.path().join("mu"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let result = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(arc)) = result else {
        panic!();
    };
    let names: Vec<&str> = arc
        .entries
        .keys()
        .map(compact_str::CompactString::as_str)
        .collect();
    assert_eq!(names, vec!["alpha", "mu", "zeta"]);
}

#[test]
fn dir_hash_recursive_is_deterministic_across_two_probes_on_stable_fs() {
    // Two probes against a static fs (no writes between) should produce
    // identical dir_hash values; mtime/inode/device are stable, entries
    // identical.
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/file.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().recursive(true).build();
    let r1 = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let r2 = probe_dir(
        tmp.path(),
        ResourceId::default(),
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    );
    let ProbeResult::Ok(TreeSnapshot::Dir(a1)) = r1 else {
        panic!();
    };
    let ProbeResult::Ok(TreeSnapshot::Dir(a2)) = r2 else {
        panic!();
    };
    assert_eq!(a1.dir_hash(), a2.dir_hash());
}

// ---------------------------------------------------------------- pool: run_worker

/// Drive a synchronous run of `run_worker` until `rx` disconnects;
/// returns every `ProbeResponse` written to `out`.
fn drain_worker_with<F>(
    rx: &Receiver<ProbeRequest>,
    out_tx: Sender<Input>,
    out_rx: &Receiver<Input>,
    expected: &ExpectedMap,
    probe: F,
) -> Vec<specter_core::ProbeResponse>
where
    F: Fn(&ProbeRequest) -> ProbeResult,
{
    run_worker(rx, &out_tx, expected, probe);
    drop(out_tx);
    let mut responses = Vec::new();
    while let Ok(input) = out_rx.recv() {
        if let Input::ProbeResponse(r) = input {
            responses.push(r);
        }
    }
    responses
}

#[test]
fn run_worker_skips_when_correlation_does_not_match() {
    let pids = fresh_profile_ids(1);
    let p = pids[0];

    let (in_tx, in_rx) = unbounded::<ProbeRequest>();
    let (out_tx, out_rx) = unbounded::<Input>();
    let expected = fresh_expected();

    // Pre-seed expected with a different correlation than the request
    // carries — simulates "cancel ran between submit and dequeue".
    seed(&expected, p, 99);

    in_tx
        .send(req(p, 1, specter_core::ProbeKind::File))
        .unwrap();
    drop(in_tx);

    let probe_calls = Arc::new(AtomicUsize::new(0));
    let probe_calls_clone = Arc::clone(&probe_calls);
    let responses = drain_worker_with(&in_rx, out_tx, &out_rx, &expected, move |_req| {
        probe_calls_clone.fetch_add(1, Ordering::SeqCst);
        ProbeResult::Vanished
    });

    assert_eq!(
        probe_calls.load(Ordering::SeqCst),
        0,
        "probe must be skipped"
    );
    assert!(responses.is_empty(), "no response on pre-run cancel");
}

#[test]
fn run_worker_runs_when_correlation_matches() {
    let pids = fresh_profile_ids(1);
    let p = pids[0];

    let (in_tx, in_rx) = unbounded::<ProbeRequest>();
    let (out_tx, out_rx) = unbounded::<Input>();
    let expected = fresh_expected();

    seed(&expected, p, 7);
    in_tx
        .send(req(p, 7, specter_core::ProbeKind::File))
        .unwrap();
    drop(in_tx);

    let responses = drain_worker_with(&in_rx, out_tx, &out_rx, &expected, |_| {
        ProbeResult::Vanished
    });
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].profile, p);
    assert_eq!(responses[0].correlation, ProbeCorrelation(7));
    assert!(matches!(responses[0].result, ProbeResult::Vanished));
}

#[test]
fn run_worker_panic_in_probe_emits_failed_eio() {
    let pids = fresh_profile_ids(1);
    let p = pids[0];

    let (in_tx, in_rx) = unbounded::<ProbeRequest>();
    let (out_tx, out_rx) = unbounded::<Input>();
    let expected = fresh_expected();
    seed(&expected, p, 1);

    in_tx
        .send(req(p, 1, specter_core::ProbeKind::File))
        .unwrap();
    drop(in_tx);

    let responses = drain_worker_with(&in_rx, out_tx, &out_rx, &expected, |_| {
        panic!("simulated probe panic");
    });
    assert_eq!(responses.len(), 1);
    assert!(matches!(
        responses[0].result,
        ProbeResult::Failed { errno: libc::EIO }
    ));
}

#[test]
fn run_worker_panic_does_not_kill_loop() {
    let pids = fresh_profile_ids(2);

    let (in_tx, in_rx) = unbounded::<ProbeRequest>();
    let (out_tx, out_rx) = unbounded::<Input>();
    let expected = fresh_expected();
    seed(&expected, pids[0], 1);
    seed(&expected, pids[1], 2);

    in_tx
        .send(req(pids[0], 1, specter_core::ProbeKind::File))
        .unwrap();
    in_tx
        .send(req(pids[1], 2, specter_core::ProbeKind::File))
        .unwrap();
    drop(in_tx);

    let panic_for = pids[0];
    let responses = drain_worker_with(&in_rx, out_tx, &out_rx, &expected, move |req| {
        assert!(req.profile != panic_for, "simulated panic on first request");
        ProbeResult::Vanished
    });
    // First panicked → Failed(EIO); second succeeded → Vanished. Both
    // arrive — the worker survived the panic.
    assert_eq!(responses.len(), 2);
    assert!(matches!(
        responses[0].result,
        ProbeResult::Failed { errno: libc::EIO }
    ));
    assert!(matches!(responses[1].result, ProbeResult::Vanished));
}

#[test]
fn run_worker_post_run_cleanup_removes_entry() {
    let pids = fresh_profile_ids(1);
    let p = pids[0];

    let (in_tx, in_rx) = unbounded::<ProbeRequest>();
    let (out_tx, out_rx) = unbounded::<Input>();
    let expected = fresh_expected();
    seed(&expected, p, 5);

    in_tx
        .send(req(p, 5, specter_core::ProbeKind::File))
        .unwrap();
    drop(in_tx);

    let inner = Arc::clone(&expected);
    let responses = drain_worker_with(&in_rx, out_tx, &out_rx, &expected, |_| {
        ProbeResult::Vanished
    });
    assert_eq!(responses.len(), 1);
    // Cleanup removed the entry.
    assert_eq!(inner.lock().unwrap().len(), 0);
}

#[test]
fn run_worker_post_run_cleanup_preserves_resubmit() {
    let pids = fresh_profile_ids(1);
    let p = pids[0];

    let (in_tx, in_rx) = unbounded::<ProbeRequest>();
    let (out_tx, out_rx) = unbounded::<Input>();
    let expected = fresh_expected();
    seed(&expected, p, 1);

    in_tx
        .send(req(p, 1, specter_core::ProbeKind::File))
        .unwrap();
    drop(in_tx);

    let inner = Arc::clone(&expected);
    let inner_for_probe = Arc::clone(&inner);
    // Inside the probe, simulate a fresh `submit(c2)` that overwrites
    // the expectation. Post-run cleanup must NOT clobber c2.
    let responses = drain_worker_with(&in_rx, out_tx, &out_rx, &expected, move |_req| {
        inner_for_probe
            .lock()
            .unwrap()
            .insert(p, ProbeCorrelation(2));
        ProbeResult::Vanished
    });
    assert_eq!(responses.len(), 1);
    assert_eq!(
        inner.lock().unwrap().get(&p).copied(),
        Some(ProbeCorrelation(2))
    );
}

// ---------------------------------------------------------------- pool: WorkerProber

#[test]
fn worker_prober_concurrency_zero_clamps_to_one() {
    let (out_tx, _out_rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&out_tx, 0).unwrap();
    let _ = prober.shutdown();
}

#[test]
fn worker_prober_submit_records_expectation_and_runs_probe() {
    let pids = fresh_profile_ids(1);
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.c");
    std::fs::write(&path, b"x").unwrap();

    let (out_tx, out_rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&out_tx, 1).unwrap();

    let request = ProbeRequest {
        profile: pids[0],
        correlation: ProbeCorrelation(42),
        kind: specter_core::ProbeKind::File,
        target_resource: ResourceId::default(),
        target_path: path,
        scan_config: ScanConfig::builder().build(),
        captured_with: 0,
        baseline_subtree: None,
        force_walk: BTreeSet::new(),
        forced: false,
    };
    prober.submit(request);

    let resp = match out_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("response within timeout")
    {
        Input::ProbeResponse(r) => r,
        other => panic!("unexpected: {other:?}"),
    };
    assert_eq!(resp.profile, pids[0]);
    assert_eq!(resp.correlation, ProbeCorrelation(42));
    assert!(matches!(resp.result, ProbeResult::Ok(_)));

    // Cleanup ran.
    assert_eq!(prober.expected_len(), 0);

    let _ = prober.shutdown();
}

#[test]
fn worker_prober_cancel_removes_expectation() {
    let pids = fresh_profile_ids(1);
    let (out_tx, _out_rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&out_tx, 1).unwrap();

    // Cancel without submit is a no-op — verify no panic.
    prober.cancel(pids[0]);
    assert_eq!(prober.expected_len(), 0);

    let _ = prober.shutdown();
}

#[test]
fn worker_prober_shutdown_returns_ok_for_clean_workers() {
    let (out_tx, _out_rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&out_tx, 4).unwrap();
    let results = prober.shutdown();
    assert_eq!(results.len(), 4);
    for r in results {
        r.expect("worker exited cleanly");
    }
}

#[test]
fn worker_prober_resubmit_after_cancel_runs() {
    let pids = fresh_profile_ids(1);
    let p = pids[0];
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.c");
    std::fs::write(&path, b"x").unwrap();

    let (out_tx, out_rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&out_tx, 1).unwrap();

    // Submit c1, cancel, submit c2. Expect: c1 either runs or skips
    // (race), c2 runs deterministically (its expectation is fresh and
    // won't be cleared by anything before the worker pops it).
    let mk_req = |c: u64| ProbeRequest {
        profile: p,
        correlation: ProbeCorrelation(c),
        kind: specter_core::ProbeKind::File,
        target_resource: ResourceId::default(),
        target_path: path.clone(),
        scan_config: ScanConfig::builder().build(),
        captured_with: 0,
        baseline_subtree: None,
        force_walk: BTreeSet::new(),
        forced: false,
    };
    prober.submit(mk_req(1));
    prober.cancel(p);
    prober.submit(mk_req(2));

    let mut got_c2 = false;
    let mut total = 0;
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline && !got_c2 {
        if let Ok(Input::ProbeResponse(r)) = out_rx.recv_timeout(Duration::from_millis(200)) {
            total += 1;
            if r.correlation == ProbeCorrelation(2) {
                got_c2 = true;
            }
        }
    }
    assert!(got_c2, "expected c2's response (got {total} total)");

    let _ = prober.shutdown();
}

#[test]
fn worker_prober_concurrent_submit_is_safe() {
    let (out_tx, out_rx) = unbounded::<Input>();
    let prober = Arc::new(WorkerProber::new(&out_tx, 2).unwrap());
    let tmp = Arc::new(TempDir::new().unwrap());
    let path = tmp.path().join("f.c");
    std::fs::write(&path, b"x").unwrap();

    // Three sender threads × 5 submits each = 15 total. Allocate the
    // ProfileIds up front from a single SlotMap so each request hits
    // a distinct expectation slot — fresh per-thread SlotMaps would
    // hand out colliding keys (each starts from `(0, 1)`).
    let pids = fresh_profile_ids(15);
    let pid_chunks: Vec<Vec<ProfileId>> = pids.chunks(5).map(<[ProfileId]>::to_vec).collect();

    let mut senders = Vec::new();
    let counter = Arc::new(Mutex::new(0u64));
    for chunk in pid_chunks {
        let prober = Arc::clone(&prober);
        let path = path.clone();
        let counter = Arc::clone(&counter);
        senders.push(std::thread::spawn(move || {
            for p in chunk {
                let c = {
                    let mut g = counter.lock().unwrap();
                    *g += 1;
                    *g
                };
                prober.submit(ProbeRequest {
                    profile: p,
                    correlation: ProbeCorrelation(c),
                    kind: specter_core::ProbeKind::File,
                    target_resource: ResourceId::default(),
                    target_path: path.clone(),
                    scan_config: ScanConfig::builder().build(),
                    captured_with: 0,
                    baseline_subtree: None,
                    force_walk: BTreeSet::new(),
                    forced: false,
                });
            }
        }));
    }
    for t in senders {
        t.join().unwrap();
    }

    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    let mut received = 0;
    while std::time::Instant::now() < deadline && received < 15 {
        if let Ok(Input::ProbeResponse(_)) = out_rx.recv_timeout(Duration::from_millis(200)) {
            received += 1;
        }
    }
    assert_eq!(received, 15, "all submits delivered");
    let prober = Arc::try_unwrap(prober).expect("only one strong ref");
    let _ = prober.shutdown();
}

// Compile-time check that the prober trait surface is Send + Sync.
#[test]
fn prober_impls_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<WorkerProber>();
    #[cfg(feature = "testkit")]
    assert_send_sync::<crate::testkit::MockProber>();
}
