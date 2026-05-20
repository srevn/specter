//! Tmp diff file lifecycle (`SPECTER_DIFF_PATH`).
//!
//! Path: `temp_dir.join("specter-{actuator_pid}-{corr:016x}.diff")`.
//! Actuator-pid is used (not the child pid; child pid isn't known until
//! after `Command::spawn`, but the env var must be set *before* spawn).
//! Correlation is hex-padded to 16 chars for stable lexicographic
//! ordering. Both `temp_dir` and `actuator_pid` are captured once at
//! actuator startup and held on
//! [`crate::pool::state::ActuatorState`] — no per-Effect `getenv` or
//! `getpid` syscall on the spawn path.
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
//!
//! # Lifecycle
//!
//! [`DiffTmpFile::create`] is atomic from the caller's perspective:
//! either the file is fully written and `sync_data`-flushed (`Ok`),
//! or no file exists on disk (the `Err` arm runs a rollback unlink
//! before returning). Callers treating `Err` as "no file to track"
//! are correct by construction.
//!
//! The handle is shared across plan steps via `Arc<DiffTmpFile>`:
//! every [`crate::pool::state::RunningJob`] /
//! [`crate::pool::state::PlanContinuation`] co-owns the Arc, and the
//! last drop — at plan terminus, after every step has reaped and
//! [`crate::pool::state::ActuatorState::terminate_plan`] has
//! returned — fires [`DiffTmpFile::drop`], which unlinks the file
//! (best-effort, ENOENT-silent). The leak-on-process-crash case
//! (no `Drop` runs on `process::exit`) is acceptable — a daemon
//! crash is rare and tmpfiles.d / periodic sweeps catch the orphan.
//!
//! # Embedded-delimiter limitation
//!
//! v1's sensor walk accepts any filename byte except `/` and NUL —
//! including `\n` and `\t`. Segments carrying these bytes corrupt both
//! this file's tab-separated format and the resolver's newline-joined
//! `SPECTER_{CREATED,DELETED,MODIFIED,RENAMED_FROM,RENAMED_TO,EXCLUDED}`
//! env vars. Operators with such filenames in watched trees should
//! parse `SPECTER_DIFF_PATH` records defensively (e.g., a
//! NUL-terminated reader) or constrain their watch roots. v2 will
//! switch to NUL-separated env vars and escape-encoded tmp records.

use specter_core::{CorrelationId, Diff, EntryRef, Rename};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Owned handle to an actuator-materialised diff tmp file.
///
/// Construction succeeds only when the file is fully written and
/// `sync_data`-flushed; on any I/O error the partially-written file
/// is rolled back (best-effort unlink) before [`Self::create`]
/// returns. The handle's `Drop` impl unlinks the file (best-effort,
/// ENOENT-silent) when the last `Arc<DiffTmpFile>` co-owner is
/// dropped — see the module docs for the per-plan lifecycle.
#[derive(Debug)]
pub(crate) struct DiffTmpFile {
    path: PathBuf,
}

impl DiffTmpFile {
    /// Allocate a path under `temp_dir` and atomically materialise
    /// the [`Diff`] into it. The file's name follows
    /// `specter-{actuator_pid}-{correlation:016x}.diff` (hex-padded
    /// correlation for stable lexicographic ordering across many
    /// concurrent Effects in the same `temp_dir`). On any I/O error
    /// during write or `sync_data`, the partial file is rolled back
    /// via a best-effort unlink before `Err` returns.
    pub(crate) fn create(
        temp_dir: &Path,
        actuator_pid: u32,
        correlation: CorrelationId,
        diff: &Diff,
    ) -> io::Result<Self> {
        let path = build_path(temp_dir, actuator_pid, correlation);
        match write_inner(&path, diff) {
            Ok(()) => Ok(Self { path }),
            Err(e) => {
                unlink_quiet(&path);
                Err(e)
            }
        }
    }

    /// Borrow the on-disk path. The returned `&Path` is valid for as
    /// long as any `Arc<Self>` co-owner of `*self` is alive — the
    /// resolver borrows for one `resolve_step` call; `Slot::running`
    /// / `Slot::plan_continue` co-own the Arc across the rest of
    /// the plan.
    pub(crate) fn path(&self) -> &Path {
        &self.path
    }
}

impl Drop for DiffTmpFile {
    fn drop(&mut self) {
        unlink_quiet(&self.path);
    }
}

/// Build the tmp path. Pure function; the caller decides whether to
/// create the file at the returned location.
fn build_path(temp_dir: &Path, actuator_pid: u32, correlation: CorrelationId) -> PathBuf {
    temp_dir.join(format!(
        "specter-{actuator_pid}-{corr:016x}.diff",
        corr = correlation.as_u64(),
    ))
}

/// Write the [`Diff`] to `path` in the tab-separated format
/// documented in this module's header. The `BufWriter` coalesces
/// the per-entry `writeln!` calls into one `write` syscall at flush
/// time.
fn write_inner(path: &Path, diff: &Diff) -> io::Result<()> {
    let f = std::fs::File::create(path)?;
    let mut buf = std::io::BufWriter::new(f);
    for e in &diff.created {
        write_entry(&mut buf, "created", e)?;
    }
    for e in &diff.deleted {
        write_entry(&mut buf, "deleted", e)?;
    }
    for e in &diff.modified {
        write_entry(&mut buf, "modified", e)?;
    }
    for r in &diff.renamed {
        write_rename(&mut buf, r)?;
    }
    // `into_inner` flushes the buffer; an IntoInnerError carries the
    // flush failure as its inner `io::Error`.
    let f = buf
        .into_inner()
        .map_err(std::io::IntoInnerError::into_error)?;
    f.sync_data()?;
    Ok(())
}

fn write_entry<W: Write>(w: &mut W, kind: &str, e: &EntryRef) -> io::Result<()> {
    writeln!(
        w,
        "{kind}\t{seg}\t{inode}",
        seg = e.segment,
        inode = e.fs_id.inode(),
    )
}

fn write_rename<W: Write>(w: &mut W, r: &Rename) -> io::Result<()> {
    writeln!(
        w,
        "renamed_from\t{seg}\t{inode}",
        seg = r.from.segment,
        inode = r.from.fs_id.inode(),
    )?;
    writeln!(
        w,
        "renamed_to\t{seg}\t{inode}",
        seg = r.to.segment,
        inode = r.to.fs_id.inode(),
    )?;
    Ok(())
}

/// Best-effort unlink. Logs at `warn` on non-`NotFound` errors;
/// ENOENT (already gone) is silent so the [`DiffTmpFile::drop`]
/// arm tolerates a concurrent external unlink.
fn unlink_quiet(path: &Path) {
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

    fn entry(seg: &str, inode: u64) -> EntryRef {
        EntryRef {
            segment: CompactString::from(seg),
            kind: EntryKind::File,
            fs_id: FsIdentity::synthetic(inode, 0),
        }
    }

    /// Pins F-LOW-2 + filename pattern: `create` MUST use its
    /// `temp_dir`, `actuator_pid`, and `correlation` arguments to
    /// build the on-disk path. A regression that reads from
    /// `std::env::temp_dir()` or `std::process::id()` would fail this
    /// test (custom temp_dir + custom pid won't appear in the
    /// resulting path).
    #[test]
    fn create_uses_provided_temp_dir_pid_and_correlation_in_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let handle =
            DiffTmpFile::create(dir.path(), 42, CorrelationId::from(0xab), &Diff::default())
                .expect("create");
        let path = handle.path();
        assert!(
            path.starts_with(dir.path()),
            "tmp file under provided temp_dir: got {} vs {}",
            path.display(),
            dir.path().display(),
        );
        let name = path
            .file_name()
            .expect("name")
            .to_string_lossy()
            .into_owned();
        assert_eq!(name, "specter-42-00000000000000ab.diff");
    }

    #[test]
    fn create_writes_all_categories_in_tab_separated_format() {
        let dir = tempfile::tempdir().expect("tempdir");
        let diff = Diff {
            created: smallvec![entry("c1", 1)],
            deleted: smallvec![entry("d1", 2)],
            modified: smallvec![entry("m1", 3)],
            renamed: smallvec![Rename {
                from: entry("from", 4),
                to: entry("to", 4),
            }],
        };
        let handle =
            DiffTmpFile::create(dir.path(), 1, CorrelationId::from(0), &diff).expect("create");
        let body = std::fs::read_to_string(handle.path()).expect("read");
        let expected = "created\tc1\t1\ndeleted\td1\t2\nmodified\tm1\t3\n\
                        renamed_from\tfrom\t4\nrenamed_to\tto\t4\n";
        assert_eq!(body, expected);
    }

    #[test]
    fn create_writes_rename_pair_consecutively() {
        let dir = tempfile::tempdir().expect("tempdir");
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
        let handle =
            DiffTmpFile::create(dir.path(), 1, CorrelationId::from(0), &diff).expect("create");
        let body = std::fs::read_to_string(handle.path()).expect("read");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 4);
        assert!(lines[0].starts_with("renamed_from\ta\t"));
        assert!(lines[1].starts_with("renamed_to\tA\t"));
        assert!(lines[2].starts_with("renamed_from\tb\t"));
        assert!(lines[3].starts_with("renamed_to\tB\t"));
    }

    #[test]
    fn create_uses_segment_as_relative_path() {
        let dir = tempfile::tempdir().expect("tempdir");
        let diff = Diff {
            created: smallvec![entry("src/sub/a.c", 7)],
            ..Default::default()
        };
        let handle =
            DiffTmpFile::create(dir.path(), 1, CorrelationId::from(0), &diff).expect("create");
        let body = std::fs::read_to_string(handle.path()).expect("read");
        assert!(body.contains("src/sub/a.c"));
        assert!(!body.contains("/abs"));
    }

    /// On any I/O failure during write, the `Err` arm must roll back
    /// the partial file before returning. Without rollback, a
    /// caller treating `Err` as "no file to track" would leak the
    /// partial. Forcing the failure: pass a `temp_dir` whose
    /// "directory" is a regular file — `File::create` returns
    /// ENOTDIR on the child path.
    #[test]
    fn create_returns_err_and_leaves_no_file_on_write_failure() {
        let dir = tempfile::tempdir().expect("tempdir");
        let blocker = dir.path().join("not_a_dir");
        std::fs::write(&blocker, "x").expect("write blocker");
        let bad_temp = blocker.as_path();
        let result = DiffTmpFile::create(bad_temp, 1, CorrelationId::from(0), &Diff::default());
        assert!(
            result.is_err(),
            "create must fail when temp_dir is a regular file",
        );
        let expected_path = bad_temp.join("specter-1-0000000000000000.diff");
        assert!(
            !expected_path.exists(),
            "no partial file left on disk after Err: {}",
            expected_path.display(),
        );
    }

    #[test]
    fn drop_unlinks_existing_file() {
        let dir = tempfile::tempdir().expect("tempdir");
        let handle =
            DiffTmpFile::create(dir.path(), 1, CorrelationId::from(0xc0), &Diff::default())
                .expect("create");
        let path = handle.path().to_path_buf();
        assert!(path.exists(), "file exists pre-drop");
        drop(handle);
        assert!(!path.exists(), "file unlinked on drop");
    }

    /// ENOENT-silent contract: a concurrent external unlink between
    /// create and drop must not panic the daemon thread. Pins the
    /// `unlink_quiet` arm in [`DiffTmpFile::drop`].
    #[test]
    fn drop_silent_when_file_already_unlinked() {
        let dir = tempfile::tempdir().expect("tempdir");
        let handle = DiffTmpFile::create(dir.path(), 1, CorrelationId::from(0), &Diff::default())
            .expect("create");
        let path = handle.path().to_path_buf();
        std::fs::remove_file(&path).expect("preremove");
        assert!(!path.exists());
        drop(handle);
    }
}
