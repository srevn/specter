//! `Path` → `OwnedFd` open primitives plus an `fstat`-driven kind probe.
//!
//! The macOS branch uses `O_EVTONLY` (Darwin-private flag for "open for
//! event monitoring only" — won't pin the file against `unlink`); the
//! FreeBSD branch uses `O_RDONLY`. Both unconditionally apply
//! `O_NOFOLLOW` — symlinks at the anchor path fail with `ELOOP` rather
//! than silently traversing. v1 has no follow-symlinks opt-in.

use specter_core::ResourceKind;
use std::ffi::CString;
use std::io::{self, Error};
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

/// Open `path` with the kqueue-friendly flag set for the current target.
/// Errors propagate verbatim (`EMFILE` / `ENFILE` / `ENOENT` / `EACCES`
/// / `ELOOP` are the FD-pressure / pending-path / symlink cases the
/// engine surfaces via `WatchOpRejected`).
pub(super) fn open_for_watch(path: &Path) -> io::Result<OwnedFd> {
    let cstr = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::other("path contains NUL byte"))?;

    #[cfg(target_os = "macos")]
    let flags = libc::O_EVTONLY | libc::O_NOFOLLOW;

    #[cfg(target_os = "freebsd")]
    let flags = libc::O_RDONLY | libc::O_NOFOLLOW;

    // SAFETY: `cstr` is a valid NUL-terminated C string for the
    // duration of the call; `flags` is a valid `O_*` bit set. `open`
    // returns a non-negative fd or -1 on error.
    #[allow(unsafe_code)]
    let raw = unsafe { libc::open(cstr.as_ptr(), flags) };
    if raw < 0 {
        return Err(Error::last_os_error());
    }
    // SAFETY: `raw >= 0` ⇒ `open` handed us a fresh fd we now own.
    #[allow(unsafe_code)]
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// Stat the open fd to determine `ResourceKind`. Used by the watcher's
/// per-resource kind cache — `NOTE_WRITE` on a Dir means structural
/// change; on a File, content modification.
pub(super) fn stat_kind(fd: &OwnedFd) -> io::Result<ResourceKind> {
    let mut s = MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `fd` is a valid open fd; `s` is a writable
    // `*mut libc::stat`. `fstat` returns 0 on success, populating `s`.
    #[allow(unsafe_code)]
    let n = unsafe { libc::fstat(fd.as_raw_fd(), s.as_mut_ptr()) };
    if n < 0 {
        return Err(Error::last_os_error());
    }
    // SAFETY: `fstat` returned 0 above, so every field of `s` is now
    // initialized.
    #[allow(unsafe_code)]
    let s = unsafe { s.assume_init() };
    let kind = match s.st_mode & libc::S_IFMT {
        libc::S_IFDIR => ResourceKind::Dir,
        libc::S_IFREG => ResourceKind::File,
        _ => ResourceKind::Unknown,
    };
    Ok(kind)
}
