//! Sibling unit tests for `prober::walk` and `prober::pool`.
//!
//! Walk tests use real `tempfile::TempDir` fixtures — no mocking the
//! filesystem. Pool tests drive `run_worker` directly with synchronous
//! crossbeam channels and a custom probe closure: that's how
//! cancellation, panic recovery, and post-run cleanup get
//! deterministic coverage without relying on multi-thread scheduling.

use super::pool::{ExpectedMap, WorkerProber, lock_expected, run_probe, run_worker};
use super::walk::{probe_anchor_file, probe_descent, probe_subtree};
use crate::Prober;
use compact_str::CompactString;
use crossbeam::channel::{Receiver, Sender, unbounded};
use slotmap::SlotMap;
use specter_core::{
    ChildEntry, DirChild, DirMeta, DirSnapshot, EntryKind, FsIdentity, GlobPattern, Input,
    LeafEntry, ProbeCorrelation, ProbeOutcome, ProbeOwner, ProbeRequest, ProfileId, ScanConfig,
};
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tempfile::TempDir;

fn fresh_expected() -> ExpectedMap {
    Arc::new(Mutex::new(BTreeMap::new()))
}

fn seed(map: &ExpectedMap, p: ProfileId, c: u64) {
    map.lock()
        .unwrap()
        .insert(ProbeOwner::Profile(p), ProbeCorrelation::from(c));
}

// ---------------------------------------------------------------- helpers

fn fresh_profile_ids(n: usize) -> Vec<ProfileId> {
    let mut sm = SlotMap::<ProfileId, ()>::with_key();
    (0..n).map(|_| sm.insert(())).collect()
}

fn req_anchor(profile: ProfileId, correlation: u64) -> ProbeRequest {
    ProbeRequest::AnchorFile {
        owner: ProbeOwner::Profile(profile),
        correlation: ProbeCorrelation::from(correlation),
        target_path: PathBuf::from("/dev/null"),
    }
}

/// Default arg pack for `probe_subtree` calls:
/// `captured_with = 0`, no baseline, empty `force_walk`, `forced = false`.
/// Use the explicit form when the test wants to exercise mtime-skip /
/// `force_walk` / forced.
fn psub(path: &std::path::Path, cfg: &ScanConfig) -> ProbeOutcome {
    probe_subtree(path, cfg, 0, None, &BTreeSet::new(), false)
}

/// Recursively collect every entry's relative path (segment from the
/// anchor) from a `ProbeOutcome::SubtreeOk(...)`. Sorted.
fn segments(outcome: &ProbeOutcome) -> Vec<String> {
    let ProbeOutcome::SubtreeOk(arc) = outcome else {
        panic!("expected SubtreeOk, got {outcome:?}");
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
        if let ChildEntry::Dir(DirChild::Covered(sub)) = child {
            collect_paths(sub, &composed, out);
        }
    }
}

// ---------------------------------------------------------------- walk: probe_anchor_file

#[test]
fn probe_anchor_file_returns_leaf_for_regular_file() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("foo.c");
    std::fs::write(&path, b"hello").unwrap();

    let outcome = probe_anchor_file(&path);
    let ProbeOutcome::AnchorOk(leaf) = outcome else {
        panic!("expected AnchorOk, got {outcome:?}");
    };
    assert_eq!(leaf.kind, EntryKind::File);
    assert_eq!(leaf.size, 5);
}

#[test]
fn probe_anchor_file_returns_vanished_for_missing_path() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("nope");
    assert!(matches!(probe_anchor_file(&path), ProbeOutcome::Vanished));
}

#[test]
fn probe_anchor_file_returns_vanished_for_directory() {
    let tmp = TempDir::new().unwrap();
    assert!(matches!(
        probe_anchor_file(tmp.path()),
        ProbeOutcome::Vanished
    ));
}

#[test]
fn probe_anchor_file_returns_vanished_for_symlink() {
    let tmp = TempDir::new().unwrap();
    let target = tmp.path().join("target.c");
    let link = tmp.path().join("link");
    std::fs::write(&target, b"x").unwrap();
    std::os::unix::fs::symlink(&target, &link).unwrap();

    // probe_anchor_file uses lstat — the symlink is the kind it sees, not the
    // target. Symlink ≠ regular file ⇒ Vanished.
    assert!(matches!(probe_anchor_file(&link), ProbeOutcome::Vanished));
}

// ---------------------------------------------------------------- walk: probe_subtree

#[test]
fn probe_subtree_returns_vanished_for_missing_path() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("nope");
    let cfg = ScanConfig::builder().build();
    assert!(matches!(psub(&path, &cfg), ProbeOutcome::Vanished));
}

#[test]
fn probe_subtree_returns_vanished_for_regular_file() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    assert!(matches!(psub(&path, &cfg), ProbeOutcome::Vanished));
}

#[test]
fn probe_subtree_empty_dir_returns_zero_entries() {
    let tmp = TempDir::new().unwrap();
    let cfg = ScanConfig::builder().build();
    let result = psub(tmp.path(), &cfg);
    let ProbeOutcome::SubtreeOk(arc) = result else {
        panic!("expected Ok(Dir)");
    };
    assert!(arc.entries.is_empty());
}

#[test]
fn probe_subtree_flat_lists_children() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a"), b"1").unwrap();
    std::fs::write(tmp.path().join("b"), b"2").unwrap();
    std::fs::write(tmp.path().join("c"), b"3").unwrap();
    let cfg = ScanConfig::builder().build();
    let result = psub(tmp.path(), &cfg);
    let segs = segments(&result);
    assert_eq!(
        segs,
        vec!["a".to_string(), "b".to_string(), "c".to_string()]
    );
}

#[test]
fn probe_subtree_recursive_collects_descendants() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/file.c"), b"x").unwrap();

    let cfg = ScanConfig::builder().recursive(true).build();
    let result = psub(tmp.path(), &cfg);
    let segs = segments(&result);
    assert_eq!(segs, vec!["sub".to_string(), "sub/file.c".to_string()]);
}

#[test]
fn probe_subtree_non_recursive_omits_grandchildren() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/file.c"), b"x").unwrap();

    let cfg = ScanConfig::builder().recursive(false).build();
    let result = psub(tmp.path(), &cfg);
    let segs = segments(&result);
    assert_eq!(segs, vec!["sub".to_string()]);
}

#[test]
fn probe_subtree_max_depth_one_excludes_grandchildren() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/grand.c"), b"x").unwrap();

    let cfg = ScanConfig::builder()
        .recursive(true)
        .max_depth(Some(1))
        .build();
    let segs = segments(&psub(tmp.path(), &cfg));
    assert_eq!(segs, vec!["sub".to_string()]);
}

#[test]
fn probe_subtree_max_depth_two_includes_grandchildren() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/grand.c"), b"x").unwrap();

    let cfg = ScanConfig::builder()
        .recursive(true)
        .max_depth(Some(2))
        .build();
    let segs = segments(&psub(tmp.path(), &cfg));
    assert_eq!(segs, vec!["sub".to_string(), "sub/grand.c".to_string()]);
}

#[test]
fn probe_subtree_max_depth_three_collects_three_levels() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir_all(tmp.path().join("a/b/c")).unwrap();
    std::fs::write(tmp.path().join("a/b/c/file.c"), b"x").unwrap();

    let cfg = ScanConfig::builder()
        .recursive(true)
        .max_depth(Some(3))
        .build();
    let segs = segments(&psub(tmp.path(), &cfg));
    // Depth 1: "a"; depth 2: "a/b"; depth 3: "a/b/c". File is at depth
    // 4 — excluded by max_depth=3.
    assert_eq!(
        segs,
        vec!["a".to_string(), "a/b".to_string(), "a/b/c".to_string()]
    );
}

#[test]
fn probe_subtree_exclude_drops_matched_entries() {
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
    let segs = segments(&psub(tmp.path(), &cfg));
    // `target/**` matches paths under target (and `target` itself with
    // globset's `**` semantics). `target/foo` is excluded.
    assert!(!segs.iter().any(|s| s.starts_with("target/")));
    assert!(segs.contains(&"src".to_string()));
    assert!(segs.contains(&"src/main.c".to_string()));
}

#[test]
fn probe_subtree_pattern_matches_files_only() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("main.c"), b"x").unwrap();
    std::fs::write(tmp.path().join("foo.txt"), b"x").unwrap();
    std::fs::create_dir(tmp.path().join("subdir")).unwrap();

    let pattern = GlobPattern::compile("**/*.c").unwrap();
    let cfg = ScanConfig::builder().pattern(pattern).build();
    let segs = segments(&psub(tmp.path(), &cfg));
    // .c files: in. .txt: out. Subdirs: in (Dir bypass).
    assert!(segs.contains(&"main.c".to_string()));
    assert!(!segs.contains(&"foo.txt".to_string()));
    assert!(segs.contains(&"subdir".to_string()));
}

#[test]
fn probe_subtree_pattern_recursive_matches_nested_files() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("src")).unwrap();
    std::fs::write(tmp.path().join("src/main.c"), b"x").unwrap();
    std::fs::write(tmp.path().join("src/foo.txt"), b"x").unwrap();

    let pattern = GlobPattern::compile("**/*.c").unwrap();
    let cfg = ScanConfig::builder()
        .recursive(true)
        .pattern(pattern)
        .build();
    let segs = segments(&psub(tmp.path(), &cfg));
    assert!(segs.contains(&"src".to_string()));
    assert!(segs.contains(&"src/main.c".to_string()));
    assert!(!segs.iter().any(|s| s.contains("foo.txt")));
}

#[test]
fn probe_subtree_hidden_false_skips_dotfiles() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join(".git")).unwrap();
    std::fs::write(tmp.path().join(".git/HEAD"), b"x").unwrap();
    std::fs::write(tmp.path().join("main.c"), b"x").unwrap();

    let cfg = ScanConfig::builder().recursive(true).hidden(false).build();
    let segs = segments(&psub(tmp.path(), &cfg));
    assert_eq!(segs, vec!["main.c".to_string()]);
}

#[test]
fn probe_subtree_hidden_true_includes_dotfiles() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join(".git")).unwrap();
    std::fs::write(tmp.path().join(".git/HEAD"), b"x").unwrap();
    std::fs::write(tmp.path().join("main.c"), b"x").unwrap();

    let cfg = ScanConfig::builder().recursive(true).hidden(true).build();
    let segs = segments(&psub(tmp.path(), &cfg));
    assert!(segs.contains(&".git".to_string()));
    assert!(segs.contains(&".git/HEAD".to_string()));
    assert!(segs.contains(&"main.c".to_string()));
}

#[test]
fn probe_subtree_does_not_descend_symlinks() {
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("target")).unwrap();
    std::fs::write(tmp.path().join("target/file.c"), b"x").unwrap();
    std::os::unix::fs::symlink(tmp.path().join("target"), tmp.path().join("link")).unwrap();

    let cfg = ScanConfig::builder().recursive(true).build();
    let segs = segments(&psub(tmp.path(), &cfg));
    // `link` is a symlink: emitted as Symlink entry but not descended
    // through. `target` is a real dir: emitted and descended.
    assert!(segs.contains(&"link".to_string()));
    assert!(segs.contains(&"target".to_string()));
    assert!(segs.contains(&"target/file.c".to_string()));
    // Critically: no `link/file.c` (would prove symlink traversal).
    assert!(!segs.iter().any(|s| s.starts_with("link/")));
}

#[test]
fn probe_subtree_emits_symlink_entry_kind() {
    let tmp = TempDir::new().unwrap();
    let target = tmp.path().join("target.c");
    std::fs::write(&target, b"x").unwrap();
    std::os::unix::fs::symlink(&target, tmp.path().join("link")).unwrap();

    let cfg = ScanConfig::builder().build();
    let result = psub(tmp.path(), &cfg);
    let ProbeOutcome::SubtreeOk(arc) = result else {
        panic!("expected Ok(Dir)");
    };
    let link_entry = arc.entries.get("link").expect("link entry");
    let ChildEntry::Leaf(l) = link_entry else {
        panic!("symlink emits as Leaf");
    };
    assert_eq!(l.kind, EntryKind::Symlink);
}

#[test]
fn probe_subtree_skips_unreadable_subdir_emits_remaining() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = TempDir::new().unwrap();
    let forbidden = tmp.path().join("forbidden");
    std::fs::create_dir(&forbidden).unwrap();
    std::fs::write(forbidden.join("inside.c"), b"x").unwrap();
    std::fs::set_permissions(&forbidden, std::fs::Permissions::from_mode(0o000)).unwrap();
    std::fs::write(tmp.path().join("ok.c"), b"x").unwrap();

    let cfg = ScanConfig::builder().recursive(true).build();
    let result = psub(tmp.path(), &cfg);

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

/// EACCES contract: `read_dir` failure on a subdir produces
/// `DirChild::Covered(empty_arc)` — covered-but-empty. The
/// `Uncovered` variant is reserved for uncovered slots (`recursive=false`,
/// `max_depth`, cross-filesystem, mid-walk `lstat`/kind-flip on the
/// subdir itself). The engine's reconcile path debug-asserts on
/// (Covered, Uncovered) | (Uncovered, Covered) coverage flips, so collapsing
/// the EACCES'd `Covered(empty)` into an `Uncovered` would route a perms-change
/// through an unreachable arm.
#[test]
fn probe_subtree_unreadable_subdir_emits_dir_child_some_empty() {
    use std::os::unix::fs::PermissionsExt;
    let tmp = TempDir::new().unwrap();
    let forbidden = tmp.path().join("forbidden");
    std::fs::create_dir(&forbidden).unwrap();
    std::fs::write(forbidden.join("inside.c"), b"x").unwrap();
    std::fs::set_permissions(&forbidden, std::fs::Permissions::from_mode(0o000)).unwrap();

    let cfg = ScanConfig::builder().recursive(true).build();
    let result = psub(tmp.path(), &cfg);

    // Restore perms before TempDir drops, else cleanup fails.
    std::fs::set_permissions(&forbidden, std::fs::Permissions::from_mode(0o755)).unwrap();

    let ProbeOutcome::SubtreeOk(arc) = result else {
        panic!("expected SubtreeOk, got {result:?}");
    };
    let forbidden_entry = arc
        .entries
        .get("forbidden")
        .expect("forbidden entry present in parent map");
    let ChildEntry::Dir(dc) = forbidden_entry else {
        panic!("forbidden must be a Dir entry, got {forbidden_entry:?}");
    };
    let sub = match dc {
        DirChild::Covered(s) => s,
        DirChild::Uncovered(_) => panic!(
            "EACCES'd subdir must emit DirChild::Covered(empty_arc), not Uncovered — \
             Uncovered is reserved for uncovered slots (config / cross-fs / \
             mid-walk lstat-or-kind failure)",
        ),
    };
    assert!(
        sub.entries.is_empty(),
        "EACCES read_dir produces an empty entries map; got {} entries",
        sub.entries.len(),
    );
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
) -> (Arc<DirSnapshot>, ProbeOutcome) {
    let first = probe_subtree(target, cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(arc) = first else {
        panic!("first probe failed");
    };
    let second = probe_subtree(target, cfg, 0, Some(&arc), &BTreeSet::new(), false);
    (arc, second)
}

#[test]
fn mtime_skip_returns_arc_clone_when_root_meta_matches() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let (baseline, second) = probe_then_reprobe_with_baseline(tmp.path(), &cfg);
    let ProbeOutcome::SubtreeOk(arc2) = second else {
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
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(baseline) = first else {
        panic!("first probe failed");
    };
    // Forge a baseline with a different mtime; walker enumerates fresh.
    let forged = Arc::new(DirSnapshot::new(
        DirMeta {
            mtime: std::time::UNIX_EPOCH,
            ..baseline.root_meta
        },
        baseline.captured_with,
        baseline.entries.clone(),
    ));
    let result = probe_subtree(tmp.path(), &cfg, 0, Some(&forged), &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(arc2) = result else {
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
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(baseline) = first else {
        panic!("first probe failed");
    };
    let forged = Arc::new(DirSnapshot::new(
        DirMeta {
            fs_id: FsIdentity {
                inode: baseline.root_meta.fs_id.inode.wrapping_add(1),
                device: baseline.root_meta.fs_id.device,
            },
            ..baseline.root_meta
        },
        baseline.captured_with,
        baseline.entries.clone(),
    ));
    let result = probe_subtree(tmp.path(), &cfg, 0, Some(&forged), &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(arc2) = result else {
        panic!("re-probe failed");
    };
    assert!(!Arc::ptr_eq(&forged, &arc2));
}

#[test]
fn mtime_skip_does_not_match_when_device_differs() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(baseline) = first else {
        panic!("first probe failed");
    };
    let forged = Arc::new(DirSnapshot::new(
        DirMeta {
            fs_id: FsIdentity {
                inode: baseline.root_meta.fs_id.inode,
                device: baseline.root_meta.fs_id.device.wrapping_add(1),
            },
            ..baseline.root_meta
        },
        baseline.captured_with,
        baseline.entries.clone(),
    ));
    let result = probe_subtree(tmp.path(), &cfg, 0, Some(&forged), &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(arc2) = result else {
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
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(baseline) = first else {
        panic!("first probe failed");
    };
    let prior_sub_arc = Arc::clone(
        baseline
            .lookup_covered_dir("sub")
            .expect("sub subtree present"),
    );
    let second = probe_subtree(
        tmp.path(),
        &cfg,
        0,
        Some(&baseline),
        &BTreeSet::new(),
        false,
    );
    let ProbeOutcome::SubtreeOk(top) = second else {
        panic!("re-probe failed");
    };
    // Top is the baseline Arc (mtime matched) ⇒ ptr_eq.
    assert!(Arc::ptr_eq(&baseline, &top));
    // The child subtree is shared transitively (it lives inside `top`).
    let new_sub_arc = Arc::clone(top.lookup_covered_dir("sub").expect("sub subtree present"));
    assert!(Arc::ptr_eq(&prior_sub_arc, &new_sub_arc));
}

// ---------------------------------------------------------------- walk: force_walk

#[test]
fn force_walk_with_path_in_set_refuses_skip() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(baseline) = first else {
        panic!("first probe failed");
    };
    // force_walk = {target_path} — refuse the skip even though mtime matches.
    let mut force = BTreeSet::new();
    force.insert(tmp.path().to_path_buf());
    let second = probe_subtree(tmp.path(), &cfg, 0, Some(&baseline), &force, false);
    let ProbeOutcome::SubtreeOk(arc2) = second else {
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
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(baseline) = first else {
        panic!("first probe failed");
    };
    // force_walk = {tmp/sub/file.c}: descendant; root must enumerate so we
    // can recurse into `sub`.
    let mut force = BTreeSet::new();
    force.insert(tmp.path().join("sub").join("file.c"));
    let second = probe_subtree(tmp.path(), &cfg, 0, Some(&baseline), &force, false);
    let ProbeOutcome::SubtreeOk(arc2) = second else {
        panic!("re-probe failed");
    };
    assert!(!Arc::ptr_eq(&baseline, &arc2));
}

#[test]
fn force_walk_with_unrelated_path_does_not_affect_skip() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(baseline) = first else {
        panic!("first probe failed");
    };
    // Path outside target's subtree — skip applies normally.
    let mut force = BTreeSet::new();
    force.insert(PathBuf::from("/totally/unrelated/path"));
    let second = probe_subtree(tmp.path(), &cfg, 0, Some(&baseline), &force, false);
    let ProbeOutcome::SubtreeOk(arc2) = second else {
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
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(baseline) = first else {
        panic!("first probe failed");
    };
    let prior_dir_c_arc = Arc::clone(baseline.lookup_covered_dir("dir_c").unwrap());
    let mut force = BTreeSet::new();
    force.insert(tmp.path().join("dir_a").join("dir_b"));
    let second = probe_subtree(tmp.path(), &cfg, 0, Some(&baseline), &force, false);
    let ProbeOutcome::SubtreeOk(top) = second else {
        panic!("re-probe failed");
    };
    // dir_c was untouched and not under a forced path; the recursion
    // should mtime-skip and reuse the baseline Arc.
    let new_dir_c_arc = Arc::clone(top.lookup_covered_dir("dir_c").unwrap());
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
    let ProbeOutcome::SubtreeOk(arc2) = second else {
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
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(baseline) = first else {
        panic!("first probe failed");
    };
    let second = probe_subtree(
        tmp.path(),
        &cfg,
        0,
        Some(&baseline),
        &BTreeSet::new(),
        true, // forced
    );
    let ProbeOutcome::SubtreeOk(arc2) = second else {
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
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(baseline) = first else {
        panic!("first probe failed");
    };
    let prior_sub_arc = Arc::clone(baseline.lookup_covered_dir("sub").unwrap());
    let second = probe_subtree(
        tmp.path(),
        &cfg,
        0,
        Some(&baseline),
        &BTreeSet::new(),
        true, // forced
    );
    let ProbeOutcome::SubtreeOk(top) = second else {
        panic!("re-probe failed");
    };
    let new_sub_arc = Arc::clone(top.lookup_covered_dir("sub").unwrap());
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
    let ProbeOutcome::SubtreeOk(arc2) = second else {
        panic!("re-probe failed");
    };
    assert!(Arc::ptr_eq(&baseline, &arc2));
}

// ---------------------------------------------------------------- walk: leaf-hash cache transfer
//
// When mtime-skip fails for a directory but a child leaf is observably
// unchanged, the walker constructs the fresh leaf via
// `LeafEntry::new_or_inherit`, which inherits the baseline leaf's
// `leaf_hash` when every identity field matches. Under eager
// construction the inherited value is byte-equal to what fresh
// computation would produce (both are pure functions of the fields),
// so the inheritance is unobservable at the hash level; these tests
// pin the integration shape and the identity-gate's negative arm — a
// baseline whose identity fields disagree must NOT leak its hash into
// the freshly-stat'd output.

/// Forge a baseline whose `a.c` entry has the contents `leaf_override`
/// (defaulting to a leaf with the same identity as the disk file).
/// `root_meta.mtime` is set to `UNIX_EPOCH` to defeat the walker's
/// top-level mtime-skip, forcing the per-child cache-transfer path to
/// run.
fn baseline_at_unix_epoch(
    real: &Arc<DirSnapshot>,
    leaf_override: Option<LeafEntry>,
) -> Arc<DirSnapshot> {
    let real_leaf = match real.entries.get("a.c").expect("fixture has a.c") {
        ChildEntry::Leaf(l) => l.clone(),
        ChildEntry::Dir(_) => panic!("a.c expected to be a leaf"),
    };
    let overlay = leaf_override.unwrap_or_else(|| {
        LeafEntry::new(
            real_leaf.kind,
            real_leaf.size,
            real_leaf.mtime,
            real_leaf.fs_id,
        )
    });
    let mut entries = real.entries.clone();
    entries.insert(CompactString::new("a.c"), ChildEntry::Leaf(overlay));
    Arc::new(DirSnapshot::new(
        DirMeta {
            mtime: std::time::UNIX_EPOCH,
            ..real.root_meta
        },
        real.captured_with,
        entries,
    ))
}

#[test]
fn cache_transfer_matches_baseline_hash_on_identity_match() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"hello").unwrap();
    let cfg = ScanConfig::builder().build();
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(real) = first else {
        panic!("first probe failed");
    };
    let baseline_leaf_hash = match real.entries.get("a.c").unwrap() {
        ChildEntry::Leaf(l) => l.leaf_hash(),
        ChildEntry::Dir(_) => panic!(),
    };

    let baseline = baseline_at_unix_epoch(&real, None);
    let result = probe_subtree(
        tmp.path(),
        &cfg,
        0,
        Some(&baseline),
        &BTreeSet::new(),
        false,
    );
    let ProbeOutcome::SubtreeOk(arc) = result else {
        panic!("re-probe failed");
    };
    let fresh = match arc.entries.get("a.c").unwrap() {
        ChildEntry::Leaf(l) => l,
        ChildEntry::Dir(_) => panic!(),
    };
    assert_eq!(
        fresh.leaf_hash(),
        baseline_leaf_hash,
        "identity-matching baseline must yield the same leaf_hash",
    );
}

#[test]
fn cache_transfer_skipped_when_leaf_identity_changes() {
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"hello").unwrap();
    let cfg = ScanConfig::builder().build();
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(real) = first else {
        panic!("first probe failed");
    };
    let real_leaf = match real.entries.get("a.c").unwrap() {
        ChildEntry::Leaf(l) => l.clone(),
        ChildEntry::Dir(_) => panic!(),
    };
    // Forge an identity-mismatched override (size differs by 1). The
    // baseline's hash differs from the canonical hash of the disk's
    // fields; the walker must NOT inherit the baseline's stale hash
    // — the freshly-stat'd leaf must report its true (real-fields)
    // hash, equal to `real_leaf.leaf_hash()` and unequal to the
    // mismatched baseline's hash.
    let mismatch = LeafEntry::new(
        real_leaf.kind,
        real_leaf.size.wrapping_add(1),
        real_leaf.mtime,
        real_leaf.fs_id,
    );
    let mismatch_hash = mismatch.leaf_hash();
    let baseline = baseline_at_unix_epoch(&real, Some(mismatch));

    let result = probe_subtree(
        tmp.path(),
        &cfg,
        0,
        Some(&baseline),
        &BTreeSet::new(),
        false,
    );
    let ProbeOutcome::SubtreeOk(arc) = result else {
        panic!("re-probe failed");
    };
    let fresh = match arc.entries.get("a.c").unwrap() {
        ChildEntry::Leaf(l) => l,
        ChildEntry::Dir(_) => panic!(),
    };
    assert_eq!(
        fresh.leaf_hash(),
        real_leaf.leaf_hash(),
        "fresh leaf must reflect the disk's real fields",
    );
    assert_ne!(
        fresh.leaf_hash(),
        mismatch_hash,
        "identity mismatch must defeat cache inheritance",
    );
}

#[test]
fn cache_transfer_threads_through_recursion() {
    // Bumping both root and sub mtimes forces enumeration at every level;
    // the deep leaf still has the right hash via the threaded baseline
    // subtree.
    let tmp = TempDir::new().unwrap();
    std::fs::create_dir(tmp.path().join("sub")).unwrap();
    std::fs::write(tmp.path().join("sub/file.c"), b"hello").unwrap();
    let cfg = ScanConfig::builder().recursive(true).build();
    let first = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(real) = first else {
        panic!("first probe failed");
    };
    let real_sub = Arc::clone(real.lookup_covered_dir("sub").unwrap());
    let real_leaf = match real_sub.entries.get("file.c").unwrap() {
        ChildEntry::Leaf(l) => l.clone(),
        ChildEntry::Dir(_) => panic!(),
    };
    let baseline_leaf_hash = real_leaf.leaf_hash();

    let mut sub_entries = real_sub.entries.clone();
    sub_entries.insert(
        CompactString::new("file.c"),
        ChildEntry::Leaf(LeafEntry::new(
            real_leaf.kind,
            real_leaf.size,
            real_leaf.mtime,
            real_leaf.fs_id,
        )),
    );
    let baseline_sub = Arc::new(DirSnapshot::new(
        DirMeta {
            mtime: std::time::UNIX_EPOCH,
            ..real_sub.root_meta
        },
        real_sub.captured_with,
        sub_entries,
    ));
    let mut root_entries = real.entries.clone();
    root_entries.insert(
        CompactString::new("sub"),
        ChildEntry::Dir(DirChild::Covered(Arc::clone(&baseline_sub))),
    );
    let baseline = Arc::new(DirSnapshot::new(
        DirMeta {
            mtime: std::time::UNIX_EPOCH,
            ..real.root_meta
        },
        real.captured_with,
        root_entries,
    ));

    let result = probe_subtree(
        tmp.path(),
        &cfg,
        0,
        Some(&baseline),
        &BTreeSet::new(),
        false,
    );
    let ProbeOutcome::SubtreeOk(top) = result else {
        panic!("re-probe failed");
    };
    let new_sub = top.lookup_covered_dir("sub").unwrap().as_ref();
    let fresh = match new_sub.entries.get("file.c").unwrap() {
        ChildEntry::Leaf(l) => l,
        ChildEntry::Dir(_) => panic!(),
    };
    assert_eq!(
        fresh.leaf_hash(),
        baseline_leaf_hash,
        "deep leaf hash must match baseline",
    );
}

// ---------------------------------------------------------------- walk: DirSnapshot construction

#[test]
fn dir_snapshot_root_meta_carries_lstat_triple() {
    use std::os::unix::fs::MetadataExt;
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let raw = std::fs::symlink_metadata(tmp.path()).unwrap();
    let result = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(arc) = result else {
        panic!("expected Ok(Dir)");
    };
    assert_eq!(arc.root_meta.fs_id.inode, raw.ino());
    assert_eq!(arc.root_meta.fs_id.device, raw.dev());
}

#[test]
fn dir_snapshot_captured_with_carries_request_value() {
    const STAMP: u64 = 0xCAFE_BABE_DEAD_BEEF;
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("a.c"), b"x").unwrap();
    let cfg = ScanConfig::builder().build();
    let result = probe_subtree(tmp.path(), &cfg, STAMP, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(arc) = result else {
        panic!();
    };
    assert_eq!(arc.captured_with, STAMP);
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
    let result = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(arc) = result else {
        panic!();
    };
    match arc.entries.get("sub").unwrap() {
        ChildEntry::Dir(dc) => {
            assert!(
                matches!(dc, DirChild::Uncovered(_)),
                "depth-1 dir uncovered (max_depth=1) ⇒ DirChild::Uncovered"
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
    let result = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(arc) = result else {
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
    let result = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(arc) = result else {
        panic!();
    };
    assert!(arc.entries.contains_key("main.c"));
    // `target/foo` excluded; `target` itself depends on glob semantics —
    // `target/**` matches paths under target, so `target` itself is in
    // (the prober_recursive integration test pins the same).
    let target_present = arc.entries.contains_key("target");
    let target_uncov = match arc.entries.get("target") {
        Some(ChildEntry::Dir(DirChild::Covered(s))) => s.entries.is_empty(),
        Some(ChildEntry::Dir(DirChild::Uncovered(_))) => true,
        _ => true,
    };
    assert!(
        !target_present || target_uncov,
        "target's subtree (if present) must be empty post-exclude"
    );
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
    let result = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(arc) = result else {
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
    let r1 = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let r2 = probe_subtree(tmp.path(), &cfg, 0, None, &BTreeSet::new(), false);
    let ProbeOutcome::SubtreeOk(a1) = r1 else {
        panic!();
    };
    let ProbeOutcome::SubtreeOk(a2) = r2 else {
        panic!();
    };
    assert_eq!(a1.dir_hash(), a2.dir_hash());
}

// ---------------------------------------------------------------- pool: run_probe dispatch
//
// `run_probe` is the variant-dispatch glue between `WorkerProber` and the
// three walker entry points. The cases below pin that each variant
// reaches the right walker, and that `Descent` honours its hardcoded
// override config (hidden=true, no exclude/pattern) regardless of any
// config baked into the request — the variant carries no `ScanConfig`.

#[test]
fn run_probe_dispatches_anchor_file_to_probe_anchor_file() {
    let pids = fresh_profile_ids(1);
    let p = pids[0];
    let tmp = TempDir::new().unwrap();
    let file_path = tmp.path().join("foo.c");
    std::fs::write(&file_path, b"hi").unwrap();

    let leaf_req = ProbeRequest::AnchorFile {
        owner: ProbeOwner::Profile(p),
        correlation: ProbeCorrelation::from(1),
        target_path: file_path,
    };
    assert!(matches!(run_probe(&leaf_req), ProbeOutcome::AnchorOk(_)));

    // AnchorFile against a directory: kind mismatch ⇒ Vanished.
    let dir_req = ProbeRequest::AnchorFile {
        owner: ProbeOwner::Profile(p),
        correlation: ProbeCorrelation::from(2),
        target_path: tmp.path().to_path_buf(),
    };
    assert!(matches!(run_probe(&dir_req), ProbeOutcome::Vanished));
}

#[test]
fn run_probe_dispatches_descent_to_probe_descent() {
    let pids = fresh_profile_ids(1);
    let p = pids[0];
    let tmp = TempDir::new().unwrap();
    std::fs::write(tmp.path().join("alpha"), b"x").unwrap();
    std::fs::write(tmp.path().join("beta"), b"x").unwrap();

    let req = ProbeRequest::Descent {
        owner: ProbeOwner::Profile(p),
        correlation: ProbeCorrelation::from(1),
        target_path: tmp.path().to_path_buf(),
    };
    let outcome = run_probe(&req);
    let ProbeOutcome::SubtreeOk(arc) = outcome else {
        panic!("expected SubtreeOk, got {outcome:?}");
    };
    // Descent enumerates one level — both children appear directly.
    assert!(arc.entries.contains_key("alpha"));
    assert!(arc.entries.contains_key("beta"));
}

#[test]
fn probe_descent_uses_hardcoded_override_config() {
    let tmp = TempDir::new().unwrap();
    // `.hidden` would be filtered by the default `hidden=false`;
    // `foo.tmp` would be filtered by an exclude/pattern. Descent's
    // hardcoded override has hidden=true with no exclude/pattern, so
    // every direct child must surface.
    std::fs::write(tmp.path().join(".hidden"), b"x").unwrap();
    std::fs::write(tmp.path().join("foo.tmp"), b"x").unwrap();
    std::fs::write(tmp.path().join("main.c"), b"x").unwrap();

    let outcome = probe_descent(tmp.path());
    let ProbeOutcome::SubtreeOk(arc) = outcome else {
        panic!("expected SubtreeOk, got {outcome:?}");
    };
    assert!(arc.entries.contains_key(".hidden"));
    assert!(arc.entries.contains_key("foo.tmp"));
    assert!(arc.entries.contains_key("main.c"));
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
    F: Fn(&ProbeRequest) -> ProbeOutcome,
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

    in_tx.send(req_anchor(p, 1)).unwrap();
    drop(in_tx);

    let probe_calls = Arc::new(AtomicUsize::new(0));
    let probe_calls_clone = Arc::clone(&probe_calls);
    let responses = drain_worker_with(&in_rx, out_tx, &out_rx, &expected, move |_req| {
        probe_calls_clone.fetch_add(1, Ordering::SeqCst);
        ProbeOutcome::Vanished
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
    in_tx.send(req_anchor(p, 7)).unwrap();
    drop(in_tx);

    let responses = drain_worker_with(&in_rx, out_tx, &out_rx, &expected, |_| {
        ProbeOutcome::Vanished
    });
    assert_eq!(responses.len(), 1);
    assert_eq!(responses[0].owner, ProbeOwner::Profile(p));
    assert_eq!(responses[0].correlation, ProbeCorrelation::from(7));
    assert!(matches!(responses[0].outcome, ProbeOutcome::Vanished));
}

#[test]
fn run_worker_panic_in_probe_emits_failed_eio() {
    let pids = fresh_profile_ids(1);
    let p = pids[0];

    let (in_tx, in_rx) = unbounded::<ProbeRequest>();
    let (out_tx, out_rx) = unbounded::<Input>();
    let expected = fresh_expected();
    seed(&expected, p, 1);

    in_tx.send(req_anchor(p, 1)).unwrap();
    drop(in_tx);

    let responses = drain_worker_with(&in_rx, out_tx, &out_rx, &expected, |_| {
        panic!("simulated probe panic");
    });
    assert_eq!(responses.len(), 1);
    assert!(matches!(
        responses[0].outcome,
        ProbeOutcome::Failed { errno: libc::EIO }
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

    in_tx.send(req_anchor(pids[0], 1)).unwrap();
    in_tx.send(req_anchor(pids[1], 2)).unwrap();
    drop(in_tx);

    let panic_for = ProbeOwner::Profile(pids[0]);
    let responses = drain_worker_with(&in_rx, out_tx, &out_rx, &expected, move |req| {
        assert!(req.owner() != panic_for, "simulated panic on first request");
        ProbeOutcome::Vanished
    });
    // First panicked → Failed(EIO); second succeeded → Vanished. Both
    // arrive — the worker survived the panic.
    assert_eq!(responses.len(), 2);
    assert!(matches!(
        responses[0].outcome,
        ProbeOutcome::Failed { errno: libc::EIO }
    ));
    assert!(matches!(responses[1].outcome, ProbeOutcome::Vanished));
}

#[test]
fn run_worker_post_run_cleanup_removes_entry() {
    let pids = fresh_profile_ids(1);
    let p = pids[0];

    let (in_tx, in_rx) = unbounded::<ProbeRequest>();
    let (out_tx, out_rx) = unbounded::<Input>();
    let expected = fresh_expected();
    seed(&expected, p, 5);

    in_tx.send(req_anchor(p, 5)).unwrap();
    drop(in_tx);

    let inner = Arc::clone(&expected);
    let responses = drain_worker_with(&in_rx, out_tx, &out_rx, &expected, |_| {
        ProbeOutcome::Vanished
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

    in_tx.send(req_anchor(p, 1)).unwrap();
    drop(in_tx);

    let inner = Arc::clone(&expected);
    let inner_for_probe = Arc::clone(&inner);
    // Inside the probe, simulate a fresh `submit(c2)` that overwrites
    // the expectation. Post-run cleanup must NOT clobber c2.
    let responses = drain_worker_with(&in_rx, out_tx, &out_rx, &expected, move |_req| {
        inner_for_probe
            .lock()
            .unwrap()
            .insert(ProbeOwner::Profile(p), ProbeCorrelation::from(2));
        ProbeOutcome::Vanished
    });
    assert_eq!(responses.len(), 1);
    assert_eq!(
        inner.lock().unwrap().get(&ProbeOwner::Profile(p)).copied(),
        Some(ProbeCorrelation::from(2))
    );
}

// ---------------------------------------------------------------- pool: lock_expected
//
// `lock_expected` is the prober's single panic-recovery primitive for
// the expectation map. The test pins poison resilience: a worker that
// panics while holding the lock must not bring down the rest of the
// pool — surviving callers re-lock the map and continue.

#[test]
fn lock_expected_recovers_from_poisoned_mutex() {
    let map: ExpectedMap = Arc::new(Mutex::new(BTreeMap::new()));
    // Poison the mutex by panicking with the guard held.
    let map_clone = Arc::clone(&map);
    let _ = std::thread::spawn(move || {
        let _guard = map_clone.lock().unwrap();
        panic!("intentional poison");
    })
    .join();
    assert!(
        map.is_poisoned(),
        "panic-in-lock must leave the mutex in poisoned state",
    );

    // The helper recovers the inner state and hands back a usable
    // guard; the caller writes through it normally.
    let pid = fresh_profile_ids(1)[0];
    {
        let mut guard = lock_expected(&map);
        guard.insert(ProbeOwner::Profile(pid), ProbeCorrelation::from(42));
    }

    // A second call observes the post-poison write — the map is
    // structurally consistent across the recovery boundary. Lift the
    // value out so the guard drops at end-of-statement.
    let recovered = lock_expected(&map).get(&ProbeOwner::Profile(pid)).copied();
    assert_eq!(recovered, Some(ProbeCorrelation::from(42)));
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

    let request = ProbeRequest::AnchorFile {
        owner: ProbeOwner::Profile(pids[0]),
        correlation: ProbeCorrelation::from(42),
        target_path: path,
    };
    prober.submit(request);

    let resp = match out_rx
        .recv_timeout(Duration::from_secs(2))
        .expect("response within timeout")
    {
        Input::ProbeResponse(r) => r,
        other => panic!("unexpected: {other:?}"),
    };
    assert_eq!(resp.owner, ProbeOwner::Profile(pids[0]));
    assert_eq!(resp.correlation, ProbeCorrelation::from(42));
    assert!(matches!(resp.outcome, ProbeOutcome::AnchorOk(_)));

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
    prober.cancel(ProbeOwner::Profile(pids[0]));
    assert_eq!(prober.expected_len(), 0);

    let _ = prober.shutdown();
}

#[test]
fn worker_prober_shutdown_returns_indexed_join_results() {
    let (out_tx, _out_rx) = unbounded::<Input>();
    let prober = WorkerProber::new(&out_tx, 4).expect("spawn 4 workers");
    let results = prober.shutdown();
    assert_eq!(results.len(), 4);
    let indices: Vec<usize> = results.iter().map(|(i, _)| *i).collect();
    assert_eq!(
        indices,
        vec![0, 1, 2, 3],
        "shutdown must hand back workers in spawn order so each index \
         lines up with its `specter-prober-N` thread name",
    );
    for (i, r) in results {
        r.unwrap_or_else(|_| panic!("worker {i} panicked during clean shutdown"));
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
    let mk_req = |c: u64| ProbeRequest::AnchorFile {
        owner: ProbeOwner::Profile(p),
        correlation: ProbeCorrelation::from(c),
        target_path: path.clone(),
    };
    prober.submit(mk_req(1));
    prober.cancel(ProbeOwner::Profile(p));
    prober.submit(mk_req(2));

    let mut got_c2 = false;
    let mut total = 0;
    let deadline = std::time::Instant::now() + Duration::from_secs(2);
    while std::time::Instant::now() < deadline && !got_c2 {
        if let Ok(Input::ProbeResponse(r)) = out_rx.recv_timeout(Duration::from_millis(200)) {
            total += 1;
            if r.correlation == ProbeCorrelation::from(2) {
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
                prober.submit(ProbeRequest::AnchorFile {
                    owner: ProbeOwner::Profile(p),
                    correlation: ProbeCorrelation::from(c),
                    target_path: path.clone(),
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
