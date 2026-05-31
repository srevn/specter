use std::fmt;
use std::path::{Component, Path, PathBuf};

/// Typed failure mode of [`canonicalize_lenient`].
///
/// Each variant maps to exactly one operator-actionable cause; the
/// validator translates them into specific
/// [`IssueKind`](crate::error::IssueKind)s rather than collapsing
/// through a single arm.
#[derive(Debug)]
pub(crate) enum PathError {
    /// Input failed `Path::is_absolute()`. Pre-normalisation rejects
    /// without touching the filesystem.
    NotAbsolute,
    /// Input's `OsStr` is empty. Pre-normalisation rejects before any
    /// further structural check.
    Empty,
    /// Input contains a `..` component (anywhere — leading, middle, or
    /// trailing). Pre-normalisation rejects before any I/O; the operator
    /// must supply a literal absolute path without parent-dir traversal.
    ///
    /// `.` is intentionally **not** rejected: `Path::components` already
    /// normalises away every `.` except leading, and a leading `.` makes
    /// the path non-absolute (caught above). The remaining accepted
    /// shape (`/foo/./bar` ≡ `/foo/bar`) is harmless — the kernel treats
    /// `.` as a no-op during canonicalisation.
    ContainsParentDir,
    /// `canonicalize` surfaced a non-`NotFound` `io::Error` — `PermissionDenied`,
    /// symlink loop, `NotADirectory`, EIO, etc. `at` carries the cursor
    /// at the point of failure (equal to the input on a top-level fail,
    /// a deeper ancestor on a mid-walk fail), and `source` preserves the
    /// raw `io::Error` for the operator-visible detail line.
    Inaccessible { at: PathBuf, source: std::io::Error },
    /// The canonicalised buffer carries one or more non-UTF-8 segments —
    /// typically via symlink resolution onto a non-UTF-8 byte path. The
    /// engine's [`specter_core::Tree::parse_attach_path`] gate requires
    /// UTF-8; the validator rejects up front rather than passing the
    /// buck. `resolved` is the canonical buffer with the offending bytes.
    NonUtf8 { resolved: PathBuf },
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAbsolute => write!(f, "path is not absolute"),
            Self::Empty => write!(f, "path is empty"),
            Self::ContainsParentDir => write!(f, "path contains `..` component"),
            Self::Inaccessible { at, source } => {
                write!(f, "`{}` is inaccessible: {source}", at.display())
            }
            Self::NonUtf8 { resolved } => {
                write!(
                    f,
                    "canonical path `{}` is not valid UTF-8",
                    resolved.display()
                )
            }
        }
    }
}

impl std::error::Error for PathError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Inaccessible { source, .. } = self {
            Some(source)
        } else {
            None
        }
    }
}

/// Canonicalize as much of `input` as exists, leaving the trailing
/// non-existent components literal. Always returns an absolute, UTF-8
/// path on success.
///
/// Internally three phases:
///
/// 1. **Structural pre-normalisation** ([`normalize_structure`]): no
///    I/O — rejects empty, non-absolute, and `..`-bearing inputs.
///    The downstream phases rely on its post-condition (every cursor
///    has a `file_name`; walk-up terminates at root).
/// 2. **Filesystem resolution** ([`resolve_filesystem`]): walks the
///    parent chain via in-place `cursor.pop()` until `canonicalize`
///    succeeds; reattaches the missing tail. Non-`NotFound` errors
///    collapse to [`PathError::Inaccessible`] carrying the cursor at
///    fault.
/// 3. **UTF-8 enforcement** ([`enforce_utf8`]): post-resolution check.
///    The engine's path gate requires UTF-8 — symlink resolution can
///    surface non-UTF-8 segments that the structural check (UTF-8 by
///    construction from TOML) can't see.
pub(crate) fn canonicalize_lenient(input: &Path) -> Result<PathBuf, PathError> {
    normalize_structure(input)?;
    let resolved = resolve_filesystem(input)?;
    enforce_utf8(resolved)
}

/// Pure structural pre-normalisation — no I/O. Rejects empty,
/// non-absolute, and `..`-bearing inputs.
///
/// Post-condition for downstream phases: any cursor derived from
/// `input` by `pop()` has a `Some` `file_name` (the only path shape
/// with `None` `file_name` after absoluteness — a trailing `..` — is
/// rejected here), and walk-up always reaches the filesystem root,
/// which canonicalises. Both invariants are load-bearing for
/// [`resolve_filesystem`]'s loop termination.
///
/// `Component::CurDir` is intentionally accepted: `Path::components`
/// normalises `.` away in every non-leading position and a leading `.`
/// fails `is_absolute()` above, so `CurDir` reaches us only in shapes
/// already vetted as harmless.
fn normalize_structure(input: &Path) -> Result<(), PathError> {
    if input.as_os_str().is_empty() {
        return Err(PathError::Empty);
    }
    if !input.is_absolute() {
        return Err(PathError::NotAbsolute);
    }
    for c in input.components() {
        match c {
            Component::ParentDir => return Err(PathError::ContainsParentDir),
            // Exhaustive remainder: forward-compat against future
            // `Component` variants. `Prefix` is Windows-only; never
            // appears on Unix targets but matched explicitly so a new
            // variant fails to compile rather than silently passes.
            Component::CurDir
            | Component::RootDir
            | Component::Normal(_)
            | Component::Prefix(_) => {}
        }
    }
    Ok(())
}

/// Walk up the parent chain until canonicalisation succeeds; reattach
/// the popped tail. `cursor` mutates in place via `pop()`; the only
/// per-iteration allocation is the popped `OsString` pushed onto `tail`.
///
/// Pre-normalisation guarantees:
/// - `cursor.file_name()` returns `Some` until `pop()` returns `false`.
/// - `pop()` returns `false` only at the filesystem root, which always
///   canonicalises — so the loop terminates without ever needing the
///   defensive fallback below.
fn resolve_filesystem(input: &Path) -> Result<PathBuf, PathError> {
    let mut cursor = input.to_owned();
    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    loop {
        match std::fs::canonicalize(&cursor) {
            Ok(mut canon) => {
                for seg in tail.iter().rev() {
                    canon.push(seg);
                }
                return Ok(canon);
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let name = cursor
                    .file_name()
                    .expect("pre-normalisation: cursor has a file_name until pop returns false")
                    .to_owned();
                tail.push(name);
                if !cursor.pop() {
                    // Unreachable on Unix post-normalisation: `pop()`
                    // returns false only at the filesystem root, and
                    // the root always canonicalises. Defend the release
                    // build with a typed error rather than panicking.
                    debug_assert!(false, "walk-up reached root via NotFound chain");
                    return Err(PathError::Inaccessible {
                        at: cursor,
                        source: std::io::Error::new(
                            std::io::ErrorKind::NotFound,
                            "no canonical ancestor",
                        ),
                    });
                }
            }
            Err(source) => return Err(PathError::Inaccessible { at: cursor, source }),
        }
    }
}

/// Post-resolution UTF-8 check. Symlink resolution can surface a
/// non-UTF-8 segment that the engine's
/// [`specter_core::Tree::parse_attach_path`] gate would reject; catch
/// it here so the validator's contract ("ok ⇒ engine accepts the
/// path") holds. Original TOML input is UTF-8 by construction, so the
/// only way this fails is through symlink resolution.
fn enforce_utf8(p: PathBuf) -> Result<PathBuf, PathError> {
    if p.to_str().is_some() {
        Ok(p)
    } else {
        Err(PathError::NonUtf8 { resolved: p })
    }
}

#[cfg(test)]
mod tests {
    use super::{PathError, canonicalize_lenient};
    use std::path::{Path, PathBuf};

    fn canon_tempdir(td: &tempfile::TempDir) -> PathBuf {
        td.path().canonicalize().expect("tempdir canonicalizes")
    }

    #[test]
    fn relative_path_rejected() {
        let err = canonicalize_lenient(Path::new("relative/path")).unwrap_err();
        assert!(matches!(err, PathError::NotAbsolute));
    }

    #[test]
    fn empty_path_rejected() {
        let err = canonicalize_lenient(Path::new("")).unwrap_err();
        assert!(matches!(err, PathError::Empty));
    }

    #[test]
    fn tilde_path_rejected_as_non_absolute() {
        let err = canonicalize_lenient(Path::new("~/foo")).unwrap_err();
        assert!(matches!(err, PathError::NotAbsolute));
    }

    #[test]
    fn parent_dir_component_rejected() {
        let err = canonicalize_lenient(Path::new("/srv/missing/..")).unwrap_err();
        assert!(matches!(err, PathError::ContainsParentDir), "got {err:?}");
    }

    /// `.` is intentionally accepted by pre-norm because `Path::components`
    /// normalises it away in every non-leading position. The kernel
    /// canonicalises the input as if the `.` were absent.
    #[test]
    fn cur_dir_component_silently_normalized() {
        let td = tempfile::tempdir().unwrap();
        let canon = canon_tempdir(&td);
        let with_dot = td.path().join(".").join("missing-leaf");
        let result = canonicalize_lenient(&with_dot).unwrap();
        assert_eq!(result, canon.join("missing-leaf"));
    }

    #[test]
    fn existing_directory_canonicalizes() {
        let td = tempfile::tempdir().unwrap();
        let canon = canon_tempdir(&td);
        let result = canonicalize_lenient(td.path()).unwrap();
        assert_eq!(result, canon);
    }

    #[test]
    fn missing_leaf_in_existing_parent() {
        let td = tempfile::tempdir().unwrap();
        let canon = canon_tempdir(&td);
        let leaf = td.path().join("does-not-exist");
        let result = canonicalize_lenient(&leaf).unwrap();
        assert_eq!(result, canon.join("does-not-exist"));
    }

    #[test]
    fn deeply_pending_path_walks_up_to_existing_ancestor() {
        let td = tempfile::tempdir().unwrap();
        let canon = canon_tempdir(&td);
        let pending = td.path().join("a").join("b").join("c").join("leaf.txt");
        let result = canonicalize_lenient(&pending).unwrap();
        assert_eq!(result, canon.join("a").join("b").join("c").join("leaf.txt"));
    }

    #[test]
    fn root_path_canonicalizes() {
        let result = canonicalize_lenient(Path::new("/")).unwrap();
        assert_eq!(result, Path::new("/"));
    }

    #[cfg(unix)]
    #[test]
    fn pending_under_root_uses_root_canonical() {
        let result = canonicalize_lenient(Path::new("/this-component-does-not-exist-xyz")).unwrap();
        assert_eq!(result, Path::new("/this-component-does-not-exist-xyz"));
    }

    #[cfg(unix)]
    #[test]
    fn symlinked_existing_prefix_resolves_through() {
        use std::os::unix::fs::symlink;

        let td = tempfile::tempdir().unwrap();
        let canon = canon_tempdir(&td);
        let target_dir = canon.join("real");
        std::fs::create_dir(&target_dir).unwrap();
        let link = canon.join("link");
        symlink(&target_dir, &link).unwrap();

        let pending = link.join("nope").join("leaf");
        let result = canonicalize_lenient(&pending).unwrap();
        assert_eq!(result, target_dir.join("nope").join("leaf"));
    }

    #[cfg(unix)]
    #[test]
    fn broken_symlink_leaf_treated_as_pending() {
        use std::os::unix::fs::symlink;

        let td = tempfile::tempdir().unwrap();
        let canon = canon_tempdir(&td);
        let link = canon.join("dangling");
        symlink(canon.join("missing-target"), &link).unwrap();

        let result = canonicalize_lenient(&link).unwrap();
        assert_eq!(result, canon.join("dangling"));
    }

    #[cfg(unix)]
    #[test]
    fn dangling_symlink_midpath_treated_as_pending() {
        use std::os::unix::fs::symlink;

        let td = tempfile::tempdir().unwrap();
        let canon = canon_tempdir(&td);
        let dangling = canon.join("dangling");
        symlink(canon.join("never-exists"), &dangling).unwrap();
        let pending = dangling.join("leaf");

        // Walk-up pops past the dangling symlink (which fails NotFound
        // because its target doesn't exist) and reattaches the literal
        // tail under the existing ancestor.
        let result = canonicalize_lenient(&pending).unwrap();
        assert!(
            result.ends_with(Path::new("dangling/leaf")),
            "got {}",
            result.display(),
        );
        assert!(result.is_absolute(), "got {}", result.display());
    }

    #[cfg(unix)]
    #[test]
    fn non_directory_traversal_rejected_as_inaccessible() {
        // `/regular-file/missing` triggers ENOTDIR — a non-`NotFound`
        // `io::Error` — which collapses to `Inaccessible`. Exercises the
        // arm that maps any non-`NotFound` I/O error to `Inaccessible`,
        // without root-skip gymnastics. EACCES via `chmod 0` follows the
        // same code path; verified manually (we can't drop privileges
        // inside a test process).
        let td = tempfile::tempdir().unwrap();
        let canon = canon_tempdir(&td);
        let file = canon.join("regular-file");
        std::fs::write(&file, b"hi").unwrap();
        let bad = file.join("nonexistent-child");

        let err = canonicalize_lenient(&bad).unwrap_err();
        assert!(matches!(err, PathError::Inaccessible { .. }), "got {err:?}");
    }

    /// Non-UTF-8 must surface via the `enforce_utf8` arm. Linux-only:
    /// ext4 / tmpfs accept raw bytes in filenames; APFS rejects
    /// non-UTF-8 at the FS layer, so we can't construct the scenario on
    /// macOS.
    #[cfg(target_os = "linux")]
    #[test]
    fn non_utf8_canonical_result_rejected() {
        use std::os::unix::ffi::OsStrExt;
        use std::os::unix::fs::symlink;

        let td = tempfile::tempdir().unwrap();
        let canon = canon_tempdir(&td);
        let bad_name = std::ffi::OsStr::from_bytes(b"non\xffutf8");
        let bad_dir = canon.join(bad_name);
        std::fs::create_dir(&bad_dir).expect("non-UTF-8 dir create");
        let link = canon.join("link");
        symlink(&bad_dir, &link).expect("symlink to non-UTF-8 target");

        let err = canonicalize_lenient(&link).unwrap_err();
        assert!(matches!(err, PathError::NonUtf8 { .. }), "got {err:?}");
    }
}
