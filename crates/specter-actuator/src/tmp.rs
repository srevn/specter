//! Tmp diff file lifecycle (`SPECTER_DIFF_PATH`).
//!
//! Path: `std::env::temp_dir().join("specter-{actuator_pid}-{corr:016x}.diff")`.
//! Actuator-pid is used (not the child pid; child pid isn't known until
//! after `Command::spawn`, but the env var must be set *before* spawn).
//! Correlation is hex-padded to 16 chars for stable lexicographic
//! ordering.
//!
//! Format (one entry per line, tab-separated, in this order):
//!
//! ```text
//! created<TAB><relative-path><TAB><inode>
//! deleted<TAB><relative-path><TAB><inode>
//! modified<TAB><relative-path><TAB><inode>
//! renamed_from<TAB><old-rel-path><TAB><inode>
//! renamed_to<TAB><new-rel-path><TAB><inode>
//! ```
//!
//! Each rename emits two consecutive lines (same inode in both, since
//! renames preserve inode in v1's single-mount probes). Path is
//! anchor-relative (`EntryRef.segment`); user scripts join with
//! `$SPECTER_ANCHOR` for absolute.

use specter_core::{CorrelationId, Diff, EntryRef, Rename};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Path for an Effect's diff tmp file.
#[must_use]
pub fn tmp_path(correlation: CorrelationId) -> PathBuf {
    std::env::temp_dir().join(format!(
        "specter-{pid}-{corr:016x}.diff",
        pid = std::process::id(),
        corr = correlation.as_u64(),
    ))
}

/// Write the [`Diff`] to `path` in the tab-separated diff format.
pub fn write_diff_file(path: &Path, diff: &Diff) -> io::Result<()> {
    let mut f = std::fs::File::create(path)?;
    for e in &diff.created {
        write_entry(&mut f, "created", e)?;
    }
    for e in &diff.deleted {
        write_entry(&mut f, "deleted", e)?;
    }
    for e in &diff.modified {
        write_entry(&mut f, "modified", e)?;
    }
    for r in &diff.renamed {
        write_rename(&mut f, r)?;
    }
    f.sync_data()?;
    Ok(())
}

fn write_entry(f: &mut std::fs::File, kind: &str, e: &EntryRef) -> io::Result<()> {
    writeln!(
        f,
        "{kind}\t{seg}\t{inode}",
        seg = e.segment,
        inode = e.fs_id.inode,
    )
}

fn write_rename(f: &mut std::fs::File, r: &Rename) -> io::Result<()> {
    writeln!(
        f,
        "renamed_from\t{seg}\t{inode}",
        seg = r.from.segment,
        inode = r.from.fs_id.inode,
    )?;
    writeln!(
        f,
        "renamed_to\t{seg}\t{inode}",
        seg = r.to.segment,
        inode = r.to.fs_id.inode,
    )?;
    Ok(())
}

/// Best-effort cleanup. Logs at warn on non-NotFound errors; ENOENT
/// (already gone) is silent.
pub fn cleanup(path: &Path) {
    if let Err(e) = std::fs::remove_file(path)
        && e.kind() != io::ErrorKind::NotFound
    {
        tracing::warn!(?path, ?e, "tmp diff file cleanup failed");
    }
}

#[cfg(test)]
mod tests {
    //! Sibling unit tests for [`crate::tmp`].

    use super::*;
    use compact_str::CompactString;
    use smallvec::smallvec;
    use specter_core::{CorrelationId, Diff, EntryKind, EntryRef, FsIdentity, Rename};
    use std::io::Read;

    fn entry(seg: &str, inode: u64) -> EntryRef {
        EntryRef {
            segment: CompactString::from(seg),
            kind: EntryKind::File,
            fs_id: FsIdentity { inode, device: 0 },
        }
    }

    #[test]
    fn tmp_path_includes_pid_and_correlation() {
        let p = tmp_path(CorrelationId::from(0xab));
        let s = p.to_string_lossy();
        assert!(s.contains(&format!("specter-{}-", std::process::id())));
        assert!(s.ends_with("00000000000000ab.diff"));
    }

    #[test]
    fn write_diff_file_writes_created_lines() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("test.diff");
        let diff = Diff {
            created: smallvec![entry("a.rs", 1), entry("b.rs", 2)],
            ..Default::default()
        };
        write_diff_file(&path, &diff).expect("write");
        let mut s = String::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_string(&mut s)
            .unwrap();
        assert_eq!(s, "created\ta.rs\t1\ncreated\tb.rs\t2\n");
    }

    #[test]
    fn write_diff_file_writes_all_categories_in_order() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("d.diff");
        let diff = Diff {
            created: smallvec![entry("c1", 1)],
            deleted: smallvec![entry("d1", 2)],
            modified: smallvec![entry("m1", 3)],
            renamed: smallvec![Rename {
                from: entry("from", 4),
                to: entry("to", 4),
            }],
        };
        write_diff_file(&path, &diff).expect("write");
        let mut s = String::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_string(&mut s)
            .unwrap();
        let expected = "created\tc1\t1\ndeleted\td1\t2\nmodified\tm1\t3\nrenamed_from\tfrom\t4\nrenamed_to\tto\t4\n";
        assert_eq!(s, expected);
    }

    #[test]
    fn write_diff_file_writes_rename_pair_consecutively() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("r.diff");
        let diff = Diff {
            renamed: smallvec![
                Rename {
                    from: entry("a", 1),
                    to: entry("A", 1),
                },
                Rename {
                    from: entry("b", 2),
                    to: entry("B", 2),
                },
            ],
            ..Default::default()
        };
        write_diff_file(&path, &diff).expect("write");
        let mut s = String::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_string(&mut s)
            .unwrap();
        let lines: Vec<&str> = s.lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with("renamed_from\ta\t"));
        assert!(lines[1].starts_with("renamed_to\tA\t"));
        assert!(lines[2].starts_with("renamed_from\tb\t"));
        assert!(lines[3].starts_with("renamed_to\tB\t"));
    }

    #[test]
    fn write_diff_file_uses_segment_as_relative_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("rel.diff");
        let diff = Diff {
            created: smallvec![entry("src/sub/a.c", 7)],
            ..Default::default()
        };
        write_diff_file(&path, &diff).expect("write");
        let mut s = String::new();
        std::fs::File::open(&path)
            .unwrap()
            .read_to_string(&mut s)
            .unwrap();
        assert!(s.contains("src/sub/a.c"));
        assert!(!s.contains("/abs"));
    }

    #[test]
    fn cleanup_silent_on_already_gone() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("never_existed.diff");
        cleanup(&path); // no panic
    }

    #[test]
    fn cleanup_removes_existing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = dir.path().join("exists.diff");
        std::fs::write(&path, "x").unwrap();
        assert!(path.exists());
        cleanup(&path);
        assert!(!path.exists());
    }
}
