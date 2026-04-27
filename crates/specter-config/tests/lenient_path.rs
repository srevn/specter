//! Integration tests: tempdir-backed lenient canonicalization.

use specter_config::{PathError, canonicalize_lenient};
use std::path::Path;

#[test]
fn deeply_nested_pending_under_existing_root() {
    let td = tempfile::tempdir().unwrap();
    let canon = td.path().canonicalize().unwrap();
    let pending = td.path().join("a").join("b").join("c").join("file.txt");
    let result = canonicalize_lenient(&pending).unwrap();
    assert_eq!(result, canon.join("a").join("b").join("c").join("file.txt"),);
}

#[test]
fn single_pending_segment_under_existing_root() {
    let td = tempfile::tempdir().unwrap();
    let canon = td.path().canonicalize().unwrap();
    let pending = td.path().join("solo");
    let result = canonicalize_lenient(&pending).unwrap();
    assert_eq!(result, canon.join("solo"));
}

#[cfg(unix)]
#[test]
fn symlink_to_existing_target_resolves_through_for_pending_leaf() {
    use std::os::unix::fs::symlink;

    let td = tempfile::tempdir().unwrap();
    let canon = td.path().canonicalize().unwrap();
    let target = canon.join("real-dir");
    std::fs::create_dir(&target).unwrap();
    let link = canon.join("symlink");
    symlink(&target, &link).unwrap();

    let pending = link.join("missing-leaf");
    let result = canonicalize_lenient(&pending).unwrap();
    assert_eq!(result, target.join("missing-leaf"));
}

#[cfg(unix)]
#[test]
fn symlink_to_existing_directory_canonicalizes_through() {
    use std::os::unix::fs::symlink;

    let td = tempfile::tempdir().unwrap();
    let canon = td.path().canonicalize().unwrap();
    let target = canon.join("real");
    std::fs::create_dir(&target).unwrap();
    std::fs::write(target.join("leaf.txt"), b"hi").unwrap();
    let link = canon.join("link");
    symlink(&target, &link).unwrap();

    let result = canonicalize_lenient(&link.join("leaf.txt")).unwrap();
    assert_eq!(result, target.join("leaf.txt"));
}

#[test]
fn relative_path_rejected() {
    let err = canonicalize_lenient(Path::new("relative")).unwrap_err();
    assert!(matches!(err, PathError::NotAbsolute));
}

#[test]
fn empty_path_rejected() {
    let err = canonicalize_lenient(Path::new("")).unwrap_err();
    assert!(matches!(err, PathError::Empty));
}
