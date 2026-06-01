//! UNIX-socket path resolution, atomic-rename binding, stale-socket
//! recovery, and the drop-guard that unlinks the socket on graceful
//! shutdown or panic.
//!
//! # Atomic-rename binding
//!
//! The bind sequence is `bind → chmod → rename`: a private staging
//! name takes the listener, gets its permissions set, then is moved
//! onto the well-known socket path with POSIX `rename(2)`'s
//! same-directory atomicity. Any process racing on the well-known
//! name only sees the listener after both the bind AND the chmod
//! have completed — there is no observation window where the
//! operator-visible path exists at a more-permissive mode.
//!
//! No `unsafe`, no `libc`. [`std::os::unix::fs::PermissionsExt`]
//! and [`std::fs::rename`] do the load-bearing work.
//!
//! # Visibility
//!
//! Every export is `pub(crate)`. `App::run` wires every one (parent
//! pre-check, stale-or-remove check, atomic bind, disarm at graceful
//! shutdown). The committed path itself comes from [`super::resolve`].

use std::fmt;
use std::fs;
use std::io::{self, ErrorKind};
use std::os::unix::fs::{FileTypeExt, PermissionsExt};
use std::os::unix::net::{UnixListener, UnixStream};
use std::path::{Path, PathBuf};

use super::resolve::{SocketCandidate, SocketSource};

/// File-mode bits applied to the bound socket. `0o600` is owner
/// read/write only — defense-in-depth on every supported deployment.
const SOCKET_MODE: u32 = 0o600;

/// Bind a [`UnixListener`] at `path` with `0o600` permissions using
/// atomic-rename: bind to a private staging name, chmod, then
/// POSIX-`rename(2)` onto the well-known path.
///
/// The well-known name is never observable at a more-permissive
/// mode — the operator-facing path appears only after the chmod has
/// already run on the staging entry. A guard wraps the staging file
/// so any post-bind failure (chmod, rename) cleans up the leaked
/// entry rather than leaving it behind for the next boot to inherit.
///
/// On success, the returned [`UnlinkGuard`] removes the bound
/// socket from disk when dropped (panic) or when
/// [`UnlinkGuard::unlink_now`] is called as part of graceful shutdown.
///
/// # Single-instance assumption
///
/// `bind_socket_atomic` does NOT defend against two daemons starting
/// within microseconds of each other against the same `path`. The
/// caller's [`check_stale_or_remove`] → this fn's `bind` → `chmod` →
/// `rename` sequence has no kernel-level mutex; a parallel pair can
/// both pass the staleness check, both bind at distinct staging
/// names, and both rename onto `path`. The rename is atomic per
/// POSIX, so one wins; the loser's [`UnlinkGuard`] (still armed
/// against the well-known `path`) wipes the *winner's* socket when
/// the loser shuts down. Operators connecting after the loser's
/// shutdown see `ENOENT` against a daemon that is still running on
/// the orphaned-inode listen queue.
///
/// For single-user alpha this race is operator-discipline-bounded
/// (one daemon per host). The structural fix is a sibling lockfile
/// (`fs2::FileExt::try_lock_exclusive` or `rustix::fs::flock`) held
/// for the daemon's lifetime — deferred until the
/// multi-daemon-per-host scenario is required.
pub(crate) fn bind_socket_atomic(path: &Path) -> io::Result<(UnixListener, UnlinkGuard)> {
    let tmp = temp_sibling(path);
    let _ = fs::remove_file(&tmp);

    let listener = UnixListener::bind(&tmp)?;
    finalize_atomic_rename(listener, &tmp, path).inspect_err(|_e| {
        // Chmod or rename failed: the staging file leaked. Clean up
        // so a retry (or the next boot) sees a tidy directory.
        let _ = fs::remove_file(&tmp);
    })
}

/// Run the chmod-then-rename tail of [`bind_socket_atomic`]. Split
/// out so the caller can attach uniform staging-file cleanup via
/// [`Result::map_err`] without duplicating the cleanup at each `?`
/// site.
fn finalize_atomic_rename(
    listener: UnixListener,
    tmp: &Path,
    path: &Path,
) -> io::Result<(UnixListener, UnlinkGuard)> {
    let mut perms = fs::metadata(tmp)?.permissions();
    perms.set_mode(SOCKET_MODE);
    fs::set_permissions(tmp, perms)?;
    fs::rename(tmp, path)?;
    Ok((listener, UnlinkGuard::new(path.to_owned())))
}

/// Fixed prefix of the staging suffix [`temp_sibling`] appends, ahead
/// of the PID. Single-sourced here so [`STAGING_SUFFIX_MAX`] and the
/// live format share one literal and cannot drift apart.
const STAGING_PREFIX: &str = ".tmp.";

/// Worst-case byte width of the staging suffix [`temp_sibling`] appends
/// ([`STAGING_PREFIX`] then the PID): the prefix's bytes + a `u32` PID's
/// 10 decimal digits. The single source of truth for the staging
/// format's width — the socket-path length budget in `crate::ipc::resolve`
/// subtracts it (plus the `sun_path` NUL) so a resolved operator override
/// still fits once [`bind_socket_atomic`] stages it. The `temp_sibling`
/// guard test pins the live suffix to this bound, turning any drift into
/// a test failure rather than a silently shrunk usable path length.
pub(crate) const STAGING_SUFFIX_MAX: usize = STAGING_PREFIX.len() + 10;

/// Construct the staging sibling name for `path`, suffixed with the
/// current PID. Built by appending to the path's `OsString` (not
/// `Path::with_extension`, which would strip the `.sock` segment
/// and produce `specter.tmp.NNN` — losing the kind hint operators
/// rely on when inspecting `lsof`/`fuser` output during incident
/// triage).
///
/// The PID suffix is a noise reducer, not a uniqueness guarantee:
/// the pre-bind `fs::remove_file` is the idempotency floor.
fn temp_sibling(path: &Path) -> PathBuf {
    let pid = std::process::id();
    let mut staging = path.as_os_str().to_owned();
    staging.push(format!("{STAGING_PREFIX}{pid}"));
    PathBuf::from(staging)
}

/// Why the daemon refused to bind because the socket's parent
/// directory is unusable. Carries the resolved [`SocketSource`] so the
/// `Display` can render the source-specific provisioning advice — this
/// is the one bind-failure mode where that advice is actionable (a
/// missing runtime dir tells the operator which deployment mechanism
/// should have created it). Stale-occupant (`AddrInUse`) and raw-`bind`
/// errors stay plain [`io::Error`]s: "create the parent directory" does
/// not apply to them, and [`crate::app::run`] source-tags those logs
/// directly instead.
#[derive(Debug)]
pub(crate) struct BindFailure {
    source: SocketSource,
    path: PathBuf,
    parent: PathBuf,
    kind: ParentKind,
}

/// How the socket's parent directory is unusable.
#[derive(Debug, Clone, Copy)]
enum ParentKind {
    /// The parent directory does not exist.
    Missing,
    /// The parent path exists but is not a directory.
    NotDir,
}

impl BindFailure {
    /// Capture the resolved candidate plus the offending parent. `parent`
    /// is passed in (not re-derived) because [`precheck_bind_parent`]
    /// already holds it from the `path.parent()` it stat'd.
    fn new(candidate: &SocketCandidate, parent: &Path, kind: ParentKind) -> Self {
        Self {
            source: candidate.source,
            path: candidate.path.clone(),
            parent: parent.to_owned(),
            kind,
        }
    }
}

impl fmt::Display for BindFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let reason = match self.kind {
            ParentKind::Missing => "does not exist",
            ParentKind::NotDir => "is not a directory",
        };
        // Layout line (which path failed, tagged by source) + the
        // parent-directory reason + the source's escape advice, each on
        // its own indented line so the operator reads cause then cure.
        write!(
            f,
            "cannot bind IPC socket at {} ({}):\n  parent directory {} {}.\n  {}",
            self.path.display(),
            self.source.label(),
            self.parent.display(),
            reason,
            self.source.daemon_hint(),
        )
    }
}

impl std::error::Error for BindFailure {}

/// Pre-bind check that the socket's parent directory exists and is a
/// directory, yielding a source-attributed [`BindFailure`] (carrying
/// actionable provisioning advice) when it is not.
///
/// Runs BEFORE [`check_stale_or_remove`] so the dominant deployment
/// failure — a system unit without its `RuntimeDirectory`, a bare
/// container with no `/run/specter` — surfaces as actionable advice
/// rather than the raw `ENOENT` [`bind_socket_atomic`] would otherwise
/// emit: a missing parent makes `check_stale_or_remove`'s
/// `symlink_metadata` return `NotFound`, which it reads as "absent,
/// proceed", so the failure would reach `bind` unannotated.
///
/// `fs::metadata` (not `symlink_metadata`) so a symlinked runtime dir
/// resolves to its target — a dangling symlink reads as `Missing`, a
/// symlink to a non-directory as `NotDir`.
///
/// TOCTOU is **error-quality only**: a parent that vanishes between this
/// check and the bind still fails safely at `bind`, only without the
/// tailored advice. Any stat error other than `NotFound` (e.g. `EACCES`
/// on an ancestor) is left for the real `bind` to surface — pre-
/// classifying every errno here would duplicate the kernel's reporting.
/// A path with no parent component (the filesystem root) has nothing to
/// pre-check and passes through.
pub(crate) fn precheck_bind_parent(candidate: &SocketCandidate) -> Result<(), BindFailure> {
    let Some(parent) = candidate.path.parent() else {
        return Ok(());
    };
    match fs::metadata(parent) {
        Ok(meta) if meta.is_dir() => Ok(()),
        Ok(_) => Err(BindFailure::new(candidate, parent, ParentKind::NotDir)),
        Err(e) if e.kind() == ErrorKind::NotFound => {
            Err(BindFailure::new(candidate, parent, ParentKind::Missing))
        }
        Err(_) => Ok(()),
    }
}

/// Probe `path` to decide whether it is a live socket (refuse to
/// bind), a stale socket / orphan non-socket file (unlink + return
/// Ok), or genuinely absent (return Ok).
///
/// Two-stage dispatch:
///
/// 1. **Stat the path.** Absent ⇒ return `Ok`. Present but not a
///    socket inode (regular file, directory entry left by a crashed
///    daemon, etc.) ⇒ unlink + return `Ok`. The stat-then-unlink
///    arm matters because path-based AF_UNIX `connect(2)` reports
///    non-socket inodes inconsistently across kernels: Linux
///    surfaces `ECONNREFUSED` (would collapse into the connect-arm),
///    macOS/BSD surface `ENOTSOCK` (an uncategorized OS error that
///    no stable [`ErrorKind`] variant covers). Checking the inode
///    type up front keeps the behaviour uniform without a
///    `cfg`-gated errno table.
/// 2. **Connect.** Only reached when the inode is in fact a socket.
///    `Ok` ⇒ live peer; abort boot with [`ErrorKind::AddrInUse`].
///    Any connect error funnels into a final unlink attempt; the
///    unlink propagates its own error, so a permission-denied
///    unlink surfaces with the precise reason an operator needs
///    to triage.
///
/// **No connect timeout.** `UnixStream::connect_timeout` does not
/// exist in stable `std`; adding `socket2` / `nix` purely to set
/// `SO_SNDTIMEO` before a single-shot probe over-engineers a path
/// whose kernel-side behavior is already effectively synchronous on
/// AF_UNIX (Linux/BSD/macOS all return immediately for success,
/// refusal, missing path, or permission failures). A consequence:
/// "live" includes "hung" — a foreign occupant that accepted the
/// connection but never services it still `connect`s OK, reads as
/// live (`AddrInUse`), and the new daemon correctly refuses rather
/// than stealing a path another process holds.
pub(crate) fn check_stale_or_remove(path: &Path) -> io::Result<()> {
    let metadata = match fs::symlink_metadata(path) {
        Ok(m) => m,
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };

    // Live socket ⇒ AddrInUse. Non-socket inode or stale socket
    // (connect refusal of any kind) ⇒ unlink and proceed; a
    // subsequent `bind` will surface anything remove_file can't
    // address. `NotFound` on the unlink collapses to `Ok` —
    // idempotent against a concurrent peer removal.
    if metadata.file_type().is_socket() && UnixStream::connect(path).is_ok() {
        return Err(io::Error::new(
            ErrorKind::AddrInUse,
            "another specter daemon already owns this socket path",
        ));
    }
    match fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == ErrorKind::NotFound => Ok(()),
        Err(e) => Err(e),
    }
}

/// Drop-guard for the bound socket file. Removes the socket from
/// disk both on graceful shutdown (via [`Self::unlink_now`]) and on
/// panic (via [`Drop`]). The guard owns the unlink responsibility
/// for the full lifetime of the bound listener.
///
/// Two states:
///
/// - **Armed** (`path` non-empty) — set on construction. `Drop`
///   runs `fs::remove_file`, so a panic anywhere between
///   `bind_socket_atomic` returning and the graceful-shutdown
///   unlink site still cleans up.
/// - **Consumed** — [`Self::unlink_now`] takes the guard by value,
///   runs the same `fs::remove_file` synchronously, and then drops
///   the internal `path` to suppress a second unlink in `Drop`.
///
/// `#[must_use]` lints away the common mistake of constructing and
/// immediately dropping the guard at a site that didn't intend
/// shutdown ordering, which would silently move the unlink earlier
/// than the IPC server thread's join.
#[derive(Debug)]
#[must_use = "UnlinkGuard removes the socket on drop; bind it to the IPC server's lifetime"]
pub(crate) struct UnlinkGuard {
    path: PathBuf,
}

impl UnlinkGuard {
    /// Wrap `path` in a fresh armed guard. Private to this module —
    /// the only producer is [`bind_socket_atomic`], so the guard's
    /// path is always one this crate has already successfully
    /// rename'd to.
    const fn new(path: PathBuf) -> Self {
        Self { path }
    }

    /// Graceful-shutdown unlink site. Performs the unlink
    /// synchronously and consumes the guard so its `Drop` is a no-op
    /// (avoids a redundant second `remove_file` call).
    ///
    /// Sole call site is [`crate::app::run`]'s shutdown sequence,
    /// after the IPC server thread has been joined — no surviving
    /// thread holds the listener fd, so removing the socket file
    /// will not break in-flight per-conn streams (each carries its
    /// own fd referring to the existing inode; the inode persists
    /// until every fd closes).
    pub(crate) fn unlink_now(mut self) {
        // Take the path out so `Drop` (still scheduled on this
        // by-value `self`) sees an empty string and skips its own
        // remove_file call.
        let path = std::mem::take(&mut self.path);
        Self::remove_quietly(&path);
    }

    /// Internal `remove_file` with the same NotFound-is-benign
    /// policy used by both the explicit `unlink_now` and the `Drop`
    /// arm. A concurrent operator `rm` or container teardown
    /// race-removing the entry is fine; any other error reaches the
    /// tracing journal so debugging surfaces the cause.
    fn remove_quietly(path: &Path) {
        if path.as_os_str().is_empty() {
            return;
        }
        if let Err(e) = fs::remove_file(path)
            && e.kind() != ErrorKind::NotFound
        {
            tracing::warn!(
                path = %path.display(),
                error = %e,
                "specter ipc: failed to unlink socket on shutdown",
            );
        }
    }
}

impl Drop for UnlinkGuard {
    fn drop(&mut self) {
        Self::remove_quietly(&self.path);
    }
}

#[cfg(test)]
mod tests {
    use super::{
        SOCKET_MODE, STAGING_SUFFIX_MAX, bind_socket_atomic, check_stale_or_remove,
        precheck_bind_parent, temp_sibling,
    };
    use crate::ipc::resolve::{SocketCandidate, SocketSource};
    use std::fs;
    use std::os::unix::fs::PermissionsExt;
    use std::path::PathBuf;

    /// A `--socket`-sourced candidate at `path`. `SocketSource::Cli`
    /// exists on every platform (the convention sources are cfg-gated),
    /// so the parent-classification tests stay platform-agnostic.
    fn cli_candidate(path: PathBuf) -> SocketCandidate {
        SocketCandidate {
            source: SocketSource::Cli,
            path,
        }
    }

    /// `bind_socket_atomic` sets exactly `0o600` on the bound file
    /// (atomic-rename + chmod ordering preserved end-to-end), and
    /// dropping the returned guard removes the path. Either property
    /// regressing alone fails the test, so the assertions are
    /// bundled into one fixture to keep the test surface narrow.
    #[test]
    fn bind_socket_atomic_sets_0600_and_unlink_guard_cleans_up() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("specter.sock");

        let (_listener, guard) =
            bind_socket_atomic(&path).expect("bind_socket_atomic on a fresh path");
        let perms = fs::metadata(&path)
            .expect("bound socket exists")
            .permissions();
        assert_eq!(
            perms.mode() & 0o777,
            SOCKET_MODE,
            "bound socket must carry the configured mode",
        );

        drop(guard);
        assert!(
            !path.exists(),
            "armed UnlinkGuard drop must remove the socket",
        );
    }

    /// `UnlinkGuard::unlink_now` is the graceful-shutdown unlink
    /// site — it removes the socket synchronously and then consumes
    /// the guard so the subsequent `Drop` doesn't try to remove the
    /// (now-missing) entry. After `unlink_now`, the path is gone.
    #[test]
    fn unlink_guard_unlink_now_removes_socket() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("specter.sock");

        let (_listener, guard) = bind_socket_atomic(&path).expect("bind on fresh path");
        guard.unlink_now();
        assert!(
            !path.exists(),
            "consumed guard must remove the socket synchronously",
        );
    }

    /// `check_stale_or_remove` removes an orphan file at the socket
    /// path (the easier-to-construct stand-in for a stale unix
    /// socket — both surface as `ConnectionRefused` on connect), and
    /// the second call returns `Ok` even though the file is already
    /// gone. Idempotency is the load-bearing property: a daemon
    /// restarting against a path it just cleaned cannot fail on the
    /// stale check just because the recovery worked.
    #[test]
    fn check_stale_or_remove_unlinks_orphan_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let path = tmp.path().join("specter.sock");
        fs::write(&path, b"stale daemon footprint").unwrap();
        assert!(path.exists());

        check_stale_or_remove(&path).expect("first call must remove the orphan");
        assert!(!path.exists(), "orphan file must be unlinked");

        check_stale_or_remove(&path).expect("second call must be a no-op on absent path");
    }

    /// `temp_sibling` appends `.tmp.<pid>` to the full path string —
    /// the staging name preserves the `.sock` kind hint (operators
    /// reading `lsof` see `specter.sock.tmp.NNN`, not `specter.tmp.NNN`).
    #[test]
    fn temp_sibling_appends_pid_suffix_to_full_basename() {
        let path = PathBuf::from("/run/user/1000/specter.sock");
        let tmp = temp_sibling(&path);
        let basename = tmp
            .file_name()
            .and_then(|s| s.to_str())
            .expect("staging path has a UTF-8 basename");
        assert!(
            basename.starts_with("specter.sock.tmp."),
            "got {basename:?}",
        );
    }

    /// `precheck_bind_parent` accepts an existing-directory parent,
    /// rejects a missing parent with a source-attributed `BindFailure`
    /// carrying the actionable hint, and rejects a non-directory parent.
    /// The three arms exercise the one new code path, so they bundle
    /// into one fixture to keep the test surface narrow.
    #[test]
    fn precheck_bind_parent_classifies_parent_directory() {
        let tmp = tempfile::TempDir::new().unwrap();

        // Existing directory parent ⇒ Ok.
        let ok = cli_candidate(tmp.path().join("specter.sock"));
        assert!(
            precheck_bind_parent(&ok).is_ok(),
            "an extant directory parent must pass the pre-check",
        );

        // Missing parent ⇒ BindFailure whose Display names the source,
        // the missing-parent reason, and the (Cli-source) creation hint.
        let missing = cli_candidate(tmp.path().join("absent").join("specter.sock"));
        let err = precheck_bind_parent(&missing).expect_err("missing parent must fail");
        let msg = err.to_string();
        assert!(
            msg.contains("from --socket")
                && msg.contains("does not exist")
                && msg.contains("create the parent directory"),
            "missing-parent failure must be source-tagged and actionable: {msg}",
        );

        // Parent path that is a file, not a directory ⇒ BindFailure.
        let file = tmp.path().join("not-a-dir");
        fs::write(&file, b"").unwrap();
        let not_dir = cli_candidate(file.join("specter.sock"));
        let err = precheck_bind_parent(&not_dir).expect_err("file parent must fail");
        assert!(
            err.to_string().contains("is not a directory"),
            "a non-directory parent must report its kind: {err}",
        );
    }

    /// The live `.tmp.<pid>` suffix `temp_sibling` appends stays within
    /// the `STAGING_SUFFIX_MAX` budget that `resolve`'s length check
    /// reserves. Pins the format to the const: widening the suffix
    /// without widening the reserve fails here rather than silently
    /// shrinking the usable socket-path length.
    #[test]
    fn temp_sibling_suffix_within_reserved_budget() {
        let base = PathBuf::from("/run/specter/specter.sock");
        let suffix = temp_sibling(&base).as_os_str().len() - base.as_os_str().len();
        assert!(
            suffix <= STAGING_SUFFIX_MAX,
            "staging suffix {suffix} exceeds reserved {STAGING_SUFFIX_MAX}",
        );
    }
}
