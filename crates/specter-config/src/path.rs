use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Debug)]
pub enum PathError {
    NotAbsolute,
    Empty,
    Io(std::io::Error),
}

impl fmt::Display for PathError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotAbsolute => write!(f, "path is not absolute"),
            Self::Empty => write!(f, "path is empty"),
            Self::Io(e) => write!(f, "io error: {e}"),
        }
    }
}

impl std::error::Error for PathError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io(e) = self {
            Some(e)
        } else {
            None
        }
    }
}

/// Canonicalize as much of `input` as exists, leaving the trailing
/// non-existent components literal. Always returns an absolute path.
///
/// Algorithm: try `canonicalize(input)`; on `NotFound`, walk up the parent
/// chain until a parent canonicalizes; reattach the missing tail. Symlinked
/// existing prefixes resolve through; pending leaves stay literal. Other
/// `io::Error` kinds (`PermissionDenied`, etc.) propagate immediately.
pub fn canonicalize_lenient(input: &Path) -> Result<PathBuf, PathError> {
    if input.as_os_str().is_empty() {
        return Err(PathError::Empty);
    }
    if !input.is_absolute() {
        return Err(PathError::NotAbsolute);
    }

    match std::fs::canonicalize(input) {
        Ok(p) => return Ok(p),
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
        Err(e) => return Err(PathError::Io(e)),
    }

    let mut tail: Vec<std::ffi::OsString> = Vec::new();
    let mut cursor: PathBuf = input.to_owned();
    loop {
        let Some(parent) = cursor.parent() else {
            return Err(PathError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "no canonical ancestor",
            )));
        };
        let Some(name) = cursor.file_name() else {
            return Err(PathError::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "path component is not a regular file name (e.g. ends in `..`)",
            )));
        };
        tail.push(name.to_owned());
        let parent = parent.to_owned();
        match std::fs::canonicalize(&parent) {
            Ok(canon_parent) => {
                let mut result = canon_parent;
                for seg in tail.iter().rev() {
                    result.push(seg);
                }
                return Ok(result);
            }
            Err(inner) if inner.kind() == std::io::ErrorKind::NotFound => {
                cursor = parent;
            }
            Err(inner) => return Err(PathError::Io(inner)),
        }
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
}
