//! Inode-level identity capture for atomic config reads + change
//! detection. Unix-only; folds the std `MetadataExt` projection into a
//! single value type without dropping out of `std` (no libc).
//!
//! Two production callsites:
//!
//! - [`Config::from_path_with_meta`](crate::Config::from_path_with_meta)
//!   captures meta atomically with the content read, via `f.metadata()`
//!   on the same `File` handle that produced the bytes. The handle pins
//!   the inode, so subsequent path-level renames (atomic-save) cannot
//!   mutate the captured value.
//! - [`FileMeta::from_path`] re-reads meta path-resolved, used by the
//!   driver's settle-expiry filter to ask "did anything substantive
//!   change?" without a full TOML parse.
//!
//! `from_path` follows symlinks (`std::fs::metadata`, not
//! `symlink_metadata`) so its inode space is the same as
//! `File::open + f.metadata()`. Lstat-only would compare the symlink's
//! own meta against the loader's stored target-inode meta — the two
//! occupy disjoint inode spaces and never agree, reload-storming the
//! driver. `metadata` also surfaces target-content edits through a
//! symlink (the lstat'd symlink's own mtime is unchanged when the
//! target is edited).

use std::fs::Metadata;
use std::os::unix::fs::MetadataExt;
use std::path::Path;

/// Inode-level identity + change-detection signal for a regular file.
///
/// Equality is structural over every field; the driver's lstat filter
/// fires a reload on any drift. `mode` / `uid` / `gid` are load-bearing
/// alongside `mtime_*`: `chmod` and `chown` move them but not `mtime`,
/// so without them a config that returns to readable after a temporary
/// `EACCES` (operator `chmod 600` → daemon lacks read → operator
/// `chmod 644`) would never recover via auto-reload — the post-recovery
/// lstat would compare equal to the pre-EACCES stored meta.
///
/// Why not `ctime`? `ctime` catches the same recovery case but also
/// moves on `setxattr` / `chflags` / `utimes` — including macOS
/// LaunchServices' `com.apple.lastuseddate#PS` xattr write on every
/// Finder open / Quick Look — which would reload-storm the driver on a
/// quiet config file. Mode + ownership is the precise filter for
/// "could this affect bytes or readability"; ctime is too coarse.
///
/// The cycle recovery further requires `loader.config_meta` to rotate
/// on **every** read attempt (success or failure), not just on success
/// — see `EngineDriver::handle_reload`'s parse-fail branch. Without
/// that, the chmod-EACCES recovery cycle is silently broken: the
/// stored meta freezes at the pre-tighten state, and the post-loosen
/// lstat compares equal to it because mode is bistable.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct FileMeta {
    pub inode: u64,
    pub device: u64,
    pub mtime_sec: i64,
    pub mtime_nsec: i64,
    pub size: u64,
    pub mode: u32,
    pub uid: u32,
    pub gid: u32,
}

impl FileMeta {
    /// Project a `std::fs::Metadata` into a `FileMeta`. The single
    /// canonical projection — used both by the atomic
    /// open+stat+read path and by [`Self::from_path`].
    #[must_use]
    pub fn from_metadata(m: &Metadata) -> Self {
        Self {
            inode: m.ino(),
            device: m.dev(),
            mtime_sec: m.mtime(),
            mtime_nsec: m.mtime_nsec(),
            size: m.len(),
            mode: m.mode(),
            uid: m.uid(),
            gid: m.gid(),
        }
    }

    /// Path-level stat (follows symlinks). Returns the captured value
    /// or the underlying `std::io::Error` on failure (ENOENT, EACCES,
    /// dangling symlink, etc.). The driver treats any error as
    /// "changed" — the subsequent reload's atomic capture either
    /// succeeds (recovery) or fails the same way (stable failure mode).
    pub fn from_path(path: &Path) -> std::io::Result<Self> {
        std::fs::metadata(path).map(|m| Self::from_metadata(&m))
    }
}

#[cfg(test)]
mod tests {
    use super::FileMeta;
    use std::os::unix::fs::PermissionsExt;
    use std::path::Path;
    use tempfile::TempDir;

    fn write(path: &Path, bytes: &[u8]) {
        std::fs::write(path, bytes).expect("write tempfile");
    }

    #[test]
    fn from_metadata_populates_every_field() {
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("a.toml");
        write(&p, b"hello");

        let meta = FileMeta::from_path(&p).unwrap();

        assert_ne!(meta.inode, 0, "inode should be non-zero on any real FS");
        assert_eq!(meta.size, 5);
        // mtime is seconds since epoch; any test running after ~1970
        // is positive. nsec sub-second component depends on FS
        // resolution (APFS/ext4 = ns-resolved; FAT-family = 0).
        assert!(meta.mtime_sec > 0, "mtime should reflect current epoch");
        // mode always carries file-type bits from the kernel; a real
        // lstat on a regular file is `S_IFREG | <perm bits>`, never 0.
        assert_ne!(meta.mode, 0, "mode should always carry file-type bits");
    }

    #[test]
    fn from_path_follows_symlinks_to_target_inode() {
        let dir = TempDir::new().unwrap();
        let target = dir.path().join("real.toml");
        let link = dir.path().join("link.toml");
        write(&target, b"hello");
        std::os::unix::fs::symlink(&target, &link).expect("symlink");

        let meta_target = FileMeta::from_path(&target).unwrap();
        let meta_link = FileMeta::from_path(&link).unwrap();

        // Symlink-follow: both paths resolve to the same inode, so the
        // captured metas are bit-equal. Lstat-only would diverge here.
        assert_eq!(
            meta_target, meta_link,
            "from_path must follow symlinks (matches File::open semantics)"
        );
    }

    #[test]
    fn from_path_propagates_io_errors() {
        let dir = TempDir::new().unwrap();
        let missing = dir.path().join("never-existed.toml");

        let err = FileMeta::from_path(&missing).unwrap_err();
        assert_eq!(err.kind(), std::io::ErrorKind::NotFound);
    }

    #[test]
    fn mode_changes_on_chmod_while_mtime_holds() {
        // Closes the chmod-after-EACCES recovery hole: a permissions
        // flip that doesn't touch content must still register as a
        // FileMeta delta, otherwise the lstat-filter would never
        // re-trigger a load attempt after the daemon was locked out.
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("a.toml");
        write(&p, b"hello");
        let mut perms = std::fs::metadata(&p).unwrap().permissions();
        perms.set_mode(0o644);
        std::fs::set_permissions(&p, perms).unwrap();

        let before = FileMeta::from_path(&p).unwrap();

        let mut perms = std::fs::metadata(&p).unwrap().permissions();
        perms.set_mode(0o600);
        std::fs::set_permissions(&p, perms).unwrap();

        let after = FileMeta::from_path(&p).unwrap();

        assert_ne!(
            before.mode, after.mode,
            "mode must move on chmod (before={:o}, after={:o})",
            before.mode, after.mode,
        );
        assert_eq!(
            (before.mtime_sec, before.mtime_nsec),
            (after.mtime_sec, after.mtime_nsec),
            "mtime must NOT move on chmod (the whole reason mode carries its weight)",
        );
    }

    #[test]
    fn fstat_on_open_handle_is_inode_pinned_across_rename() {
        // The atomicity claim of `Config::from_path_with_meta`: once
        // `File::open` binds f to inode-X, `f.metadata()` keeps
        // returning inode-X's meta regardless of subsequent
        // path-level renames. Verifies the building block
        // independently of the Config wrapper.
        //
        // Every fingerprinted field is rename-stable: rename moves the
        // path, not the inode's bytes / mtime / mode / ownership. The
        // full struct equality below pins the property end-to-end.
        let dir = TempDir::new().unwrap();
        let p = dir.path().join("a.toml");
        write(&p, b"original");

        let f = std::fs::File::open(&p).unwrap();
        let m_open = FileMeta::from_metadata(&f.metadata().unwrap());

        let backup = dir.path().join("a.toml.bak");
        std::fs::rename(&p, &backup).unwrap();
        write(&p, b"replacement-with-different-length-bytes");

        let m_after_rename = FileMeta::from_metadata(&f.metadata().unwrap());
        let m_path = FileMeta::from_path(&p).unwrap();

        assert_eq!(
            m_open, m_after_rename,
            "fstat on open fd must hold every field across rename — \
             the orphan inode's content, mtime, mode, and ownership \
             are inode-bound, not path-bound",
        );
        assert_ne!(
            m_open.inode, m_path.inode,
            "atomic-save (rename) creates a fresh inode at the path",
        );
    }
}
