//! Socket-path resolution **policy** — one decision shared by both roles, projected two ways.
//!
//! The daemon *commits* a single path to bind; the client *probes* an ordered candidate list,
//! taking the first that answers. These are not two independent resolvers: they read the **same**
//! [`Resolution`] value and differ only in how they consume it.
//!
//! - The daemon (which cannot probe — it binds exactly one path) projects via
//!   [`Resolution::into_commit`].
//! - The client (which cannot see the daemon's argv or environment) matches the variant: a
//!   [`Resolution::Pinned`] override connects to that one path; a [`Resolution::Cascade`] iterates
//!   its candidates.
//!
//! Because both roles consume one value, the rendezvous invariant — *the daemon's committed path is
//! the head of the client's probe set* — holds by construction, not by a paired equality test:
//! `into_commit()` returns the [`Cascade`](Resolution::Cascade) head, and [`Candidates::into_iter`]
//! yields that same head first.
//!
//! # Purity and the environment seam
//!
//! The policy is pure: it never touches the filesystem and reads the environment only through an
//! injected `getenv` closure, so the unit suite drives every branch with plain in-memory closures.
//! [`env_os`] is the module's single `std::env` touchpoint — the production adapter the daemon and
//! client thread in so both roles read one process environment through one definition.
//!
//! # Precedence
//!
//! `--socket` > `$SPECTER_SOCK` > the per-platform convention cascade. Explicit overrides are
//! absolute-only and length-checked here (one rule for both the flag and the env var); a relative
//! or over-long override is a hard [`ResolveError`], never a silent fallback. The fixed convention
//! paths (macOS `/tmp`, the system runtime dirs) are absolute and short by construction and skip
//! [`validate`]; the env-derived Linux session path (`$XDG_RUNTIME_DIR` joined) runs through
//! [`validate`] like an override, but an unusable one falls through to the system path instead of
//! erroring — a convention rung is a try-else-fall-back, not an operator demand.
//!
//! # Diagnostics vocabulary
//!
//! [`SocketSource`] tags every candidate with where it came from. [`SocketSource::label`] names it
//! in the client's probe-failure list; [`SocketSource::daemon_hint`] selects the daemon's
//! bind-failure escape advice ([`DaemonHint`]). Both failure surfaces render through this one vocab
//! so daemon and client stay in lockstep.

use std::ffi::OsString;
use std::fmt;
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

/// Socket filename under whichever directory the cascade selects.
const SOCKET_FILE: &str = "specter.sock";

/// Linux system-daemon socket directory. systemd provisions it `0700` and service-user-owned via
/// `RuntimeDirectory=specter`; the daemon binds here whenever no session runtime dir is present,
/// and a session client falls through to it when its own runtime dir holds no live socket.
#[cfg(target_os = "linux")]
const CONVENTION_DIR: &str = "/run/specter";

/// macOS socket directory. Fixed at `/tmp` (which resolves to `/private/tmp`): macOS has neither
/// `PrivateTmp` namespacing nor `$XDG_RUNTIME_DIR`, so one well-known location is both the daemon's
/// bind dir and the client's sole probe target — override-immune by design.
#[cfg(target_os = "macos")]
const CONVENTION_DIR: &str = "/tmp";

/// BSD system-daemon socket directory. The rc script provisions it in `start_precmd`; the daemon
/// binds here as the sole convention path.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
const CONVENTION_DIR: &str = "/var/run/specter";

/// `sockaddr_un.sun_path` capacity in bytes, including the terminating NUL. `std` does not expose
/// this, so it is cfg-gated to the platform constant: 108 on Linux, 104 on macOS and the BSDs.
#[cfg(target_os = "linux")]
const SUN_MAX: usize = 108;
#[cfg(not(target_os = "linux"))]
const SUN_MAX: usize = 104;

/// Longest explicit socket path that still fits `sun_path` once the bind-time staging suffix and
/// the terminating NUL are accounted for.
///
/// [`SUN_MAX`] (the `sun_path` capacity, NUL included) is resolve's own platform knowledge; the
/// staging suffix width is single-sourced from `sockpath::STAGING_SUFFIX_MAX`, beside the
/// `temp_sibling` code that emits the `.tmp.<pid>` format, so the reserve cannot drift from the
/// format it guards. `bind_socket_atomic` binds the staging name *before* renaming onto the committed
/// path, so the staging name — not the committed path — is the longest entry the kernel ever stores
/// in `sun_path`; reserving its worst-case width makes any path that passes [`validate`] also fit at
/// bind time. Convention paths are far shorter; this bound guards only operator-supplied overrides.
const MAX_SOCKET_PATH_LEN: usize = SUN_MAX - super::sockpath::STAGING_SUFFIX_MAX - 1;

/// Where a resolved socket path came from — drives both-side diagnostics and (for the daemon) the
/// bind-failure escape hint.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum SocketSource {
    /// `--socket <path>` on either role's command line.
    Cli,
    /// `$SPECTER_SOCK` in the process environment.
    Env,
    /// `$XDG_RUNTIME_DIR/specter.sock` — the per-user session daemon.
    #[cfg(target_os = "linux")]
    LinuxSession,
    /// `/run/specter/specter.sock` — the system daemon's runtime dir.
    #[cfg(target_os = "linux")]
    LinuxSystem,
    /// `/tmp/specter.sock` — the fixed macOS location.
    #[cfg(target_os = "macos")]
    MacosDefault,
    /// `/var/run/specter/specter.sock` — the BSD system daemon.
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    BsdSystem,
}

impl SocketSource {
    /// Short label naming this source in a client probe-failure list, e.g. `cannot reach the
    /// daemon; tried: <path> (session runtime)`.
    #[must_use]
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::Cli => "from --socket",
            Self::Env => "from SPECTER_SOCK",
            #[cfg(target_os = "linux")]
            Self::LinuxSession => "session runtime",
            #[cfg(target_os = "linux")]
            Self::LinuxSystem => "system runtime",
            #[cfg(target_os = "macos")]
            Self::MacosDefault => "default location",
            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            Self::BsdSystem => "system runtime",
        }
    }

    /// Escape advice for the daemon's bind-failure message — how the operator makes this source's
    /// parent directory exist.
    #[must_use]
    pub(crate) const fn daemon_hint(self) -> DaemonHint {
        match self {
            // An operator-named path: the operator owns provisioning it.
            Self::Cli | Self::Env => DaemonHint::OperatorProvided,
            #[cfg(target_os = "linux")]
            Self::LinuxSession => DaemonHint::SessionRuntimeDir,
            #[cfg(target_os = "linux")]
            Self::LinuxSystem => DaemonHint::SystemdRuntimeDir,
            // `/tmp` always exists; a missing parent there is an operator-environment anomaly, so
            // the generic advice fits.
            #[cfg(target_os = "macos")]
            Self::MacosDefault => DaemonHint::OperatorProvided,
            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            Self::BsdSystem => DaemonHint::BsdRcDir,
        }
    }
}

/// The source-specific escape line of a daemon bind-failure message. `Display` renders the advice
/// sentence; the caller owns the surrounding layout (which path failed, why its parent is unusable).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum DaemonHint {
    /// The path was operator-chosen (`--socket` / `$SPECTER_SOCK`) or a fixed location: its parent
    /// is the operator's to create.
    OperatorProvided,
    /// `/run/specter` is provisioned by systemd.
    #[cfg(target_os = "linux")]
    SystemdRuntimeDir,
    /// `$XDG_RUNTIME_DIR` is provided by the login session (logind).
    #[cfg(target_os = "linux")]
    SessionRuntimeDir,
    /// `/var/run/specter` is provisioned by the rc script.
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    BsdRcDir,
}

impl fmt::Display for DaemonHint {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // The universal escape, shared by every convention variant: a convention path renders its
        // source-specific provisioning clause, then "; otherwise " + this escape. An
        // operator-chosen path has no provisioning mechanism (`None` below) and renders the bare
        // creation advice alone — re-suggesting the `--socket` / `SPECTER_SOCK` the operator
        // already passed would be circular.
        const ESCAPE: &str =
            "create the parent directory, or override with --socket <path> / SPECTER_SOCK=<path>";
        let provisioning: Option<&'static str> = match self {
            Self::OperatorProvided => None,
            #[cfg(target_os = "linux")]
            Self::SystemdRuntimeDir => Some(
                "systemd provisions it via RuntimeDirectory=specter — check the unit is active",
            ),
            #[cfg(target_os = "linux")]
            Self::SessionRuntimeDir => {
                Some("your login session should provide $XDG_RUNTIME_DIR — check logind is active")
            }
            #[cfg(not(any(target_os = "linux", target_os = "macos")))]
            Self::BsdRcDir => {
                Some("the rc script provisions it in start_precmd — check the service is enabled")
            }
        };
        match provisioning {
            None => f.write_str("create the parent directory"),
            Some(clause) => write!(f, "{clause}; otherwise {ESCAPE}"),
        }
    }
}

/// One resolved socket path plus where it came from.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct SocketCandidate {
    pub(crate) source: SocketSource,
    pub(crate) path: PathBuf,
}

/// The convention cascade: a guaranteed-present `head` plus zero or more fall-through candidates.
/// Non-empty by construction (the head always exists); `tail` is a frozen `Box<[_]>` rather than a
/// `Vec` because it gains no entries after construction.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct Candidates {
    head: SocketCandidate,
    tail: Box<[SocketCandidate]>,
}

impl IntoIterator for Candidates {
    type Item = SocketCandidate;
    type IntoIter =
        std::iter::Chain<std::iter::Once<SocketCandidate>, std::vec::IntoIter<SocketCandidate>>;

    /// `head` first, then the fall-through `tail` in order — the client probe order. Owned, so the
    /// winning candidate moves out of the iterator into the live connection.
    fn into_iter(self) -> Self::IntoIter {
        std::iter::once(self.head).chain(self.tail.into_vec())
    }
}

/// The single resolution outcome both roles consume.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum Resolution {
    /// An explicit override: bind / connect to exactly this path, with no fall-through. A stale
    /// override must surface as an error, not silently retarget a different daemon.
    Pinned(SocketCandidate),
    /// The per-platform convention: the daemon binds the head; the client probes head-then-tail.
    Cascade(Candidates),
}

impl Resolution {
    /// The daemon projection — the one path to bind. Variant-agnostic: a [`Pinned`](Self::Pinned)
    /// override commits its path; a [`Cascade`](Self::Cascade) commits its head. By construction
    /// the committed path is the head of the client's probe set.
    #[must_use]
    pub(crate) fn into_commit(self) -> SocketCandidate {
        match self {
            Self::Pinned(candidate) => candidate,
            Self::Cascade(cascade) => cascade.head,
        }
    }
}

/// Why an explicit override could not be turned into a bindable path. Only an explicit override
/// (`--socket` / `$SPECTER_SOCK`) surfaces this: the env-derived Linux session path also runs
/// through [`validate`], but its failure is swallowed (`.ok()`) and falls through to the system
/// path rather than aborting resolution.
#[derive(Debug)]
pub(crate) enum ResolveError {
    /// A relative override. AF_UNIX needs an absolute path, and a relative one would silently
    /// depend on the daemon's CWD.
    Relative { source: SocketSource, path: PathBuf },
    /// An override whose byte length, plus the bind-time staging suffix, would overflow `sun_path`.
    TooLong {
        source: SocketSource,
        path: PathBuf,
        len: usize,
        limit: usize,
    },
}

impl fmt::Display for ResolveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Relative { source, path } => write!(
                f,
                "the socket path ({}) is not absolute: {} — AF_UNIX requires an absolute path",
                source.label(),
                path.display(),
            ),
            Self::TooLong {
                source,
                path,
                len,
                limit,
            } => write!(
                f,
                "the socket path ({}) is too long: {len} bytes exceeds the {limit}-byte limit \
                 (the bind-time staging suffix is reserved): {}",
                source.label(),
                path.display(),
            ),
        }
    }
}

impl std::error::Error for ResolveError {}

/// Resolve the socket path into the single [`Resolution`] both roles consume. Pure: every
/// environment read flows through `getenv`.
///
/// Precedence: `--socket` (`cli_socket`) > `$SPECTER_SOCK` > the per-platform convention cascade.
pub(crate) fn resolve<F>(cli_socket: Option<&Path>, getenv: F) -> Result<Resolution, ResolveError>
where
    F: Fn(&str) -> Option<OsString>,
{
    if let Some(candidate) = explicit_override(cli_socket, &getenv)? {
        return Ok(Resolution::Pinned(candidate));
    }
    Ok(Resolution::Cascade(convention(&getenv)))
}

/// Resolve the highest-precedence explicit override, if any. `--socket` wins over `$SPECTER_SOCK`;
/// both run through [`validate`] so the absolute-only + length rule has one home.
fn explicit_override<F>(
    cli_socket: Option<&Path>,
    getenv: &F,
) -> Result<Option<SocketCandidate>, ResolveError>
where
    F: Fn(&str) -> Option<OsString>,
{
    if let Some(path) = cli_socket {
        return validate(path, SocketSource::Cli).map(Some);
    }
    if let Some(value) = env_nonempty(getenv, "SPECTER_SOCK") {
        return validate(Path::new(&value), SocketSource::Env).map(Some);
    }
    Ok(None)
}

/// The per-platform convention cascade. Linux reads `$XDG_RUNTIME_DIR` to decide between the
/// session daemon (with the system path as a fall-through tail) and the system daemon alone; macOS
/// and BSD have a single fixed location and ignore the environment.
#[cfg(target_os = "linux")]
fn convention<F>(getenv: &F) -> Candidates
where
    F: Fn(&str) -> Option<OsString>,
{
    // The session head is env-derived (`$XDG_RUNTIME_DIR` joined with the socket filename), so it
    // runs through `validate`: a relative or over-long runtime dir yields `None` and falls through
    // to the system path rather than aborting resolution. `validate` subsumes the absolute-only
    // check, so no separate `is_absolute` filter is needed. The fixed system path is absolute and
    // short by construction and skips validation.
    match env_nonempty(getenv, "XDG_RUNTIME_DIR")
        .map(|xdg| Path::new(&xdg).join(SOCKET_FILE))
        .and_then(|path| validate(&path, SocketSource::LinuxSession).ok())
    {
        // A usable session runtime dir: prefer it, but keep the system path as a fall-through so a
        // session client still reaches a system daemon when no session daemon is running.
        Some(session) => Candidates {
            head: session,
            tail: Box::new([candidate(
                SocketSource::LinuxSystem,
                Path::new(CONVENTION_DIR).join(SOCKET_FILE),
            )]),
        },
        // No usable session runtime dir — absent, empty, relative, or an over-long join: the system
        // path is the sole convention path.
        None => single(SocketSource::LinuxSystem, CONVENTION_DIR),
    }
}

#[cfg(target_os = "macos")]
fn convention<F>(_getenv: &F) -> Candidates
where
    F: Fn(&str) -> Option<OsString>,
{
    single(SocketSource::MacosDefault, CONVENTION_DIR)
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
fn convention<F>(_getenv: &F) -> Candidates
where
    F: Fn(&str) -> Option<OsString>,
{
    single(SocketSource::BsdSystem, CONVENTION_DIR)
}

/// A single-candidate cascade rooted at `dir` (head only, empty tail).
fn single(source: SocketSource, dir: &str) -> Candidates {
    Candidates {
        head: candidate(source, Path::new(dir).join(SOCKET_FILE)),
        tail: Box::new([]),
    }
}

/// Pair a source with its path. Convention paths skip [`validate`] — they are absolute and short by
/// construction.
const fn candidate(source: SocketSource, path: PathBuf) -> SocketCandidate {
    SocketCandidate { source, path }
}

/// Enforce the absolute-only + `sun_path`-length rule on a path whose shape is not guaranteed by
/// construction — an explicit override (`--socket` / `$SPECTER_SOCK`) or the env-derived Linux
/// session path. The length budget reserves the bind-time staging suffix, so a path that passes
/// here also fits when `bind_socket_atomic` stages it.
fn validate(path: &Path, source: SocketSource) -> Result<SocketCandidate, ResolveError> {
    if !path.is_absolute() {
        return Err(ResolveError::Relative {
            source,
            path: path.to_owned(),
        });
    }
    let len = path.as_os_str().as_bytes().len();
    if len > MAX_SOCKET_PATH_LEN {
        return Err(ResolveError::TooLong {
            source,
            path: path.to_owned(),
            len,
            limit: MAX_SOCKET_PATH_LEN,
        });
    }
    Ok(candidate(source, path.to_owned()))
}

/// Read an environment variable, treating an empty value (`export X=`) as unset. The one place the
/// empty-is-unset rule lives, shared by `$SPECTER_SOCK` and `$XDG_RUNTIME_DIR`.
fn env_nonempty<F>(getenv: &F, key: &str) -> Option<OsString>
where
    F: Fn(&str) -> Option<OsString>,
{
    getenv(key).filter(|value| !value.is_empty())
}

/// Production `getenv` for [`resolve`] — the module's single `std::env` touchpoint. The daemon and
/// client thread it in so both roles read one process environment; the unit suite injects closures
/// instead.
#[must_use]
pub(crate) fn env_os(key: &str) -> Option<OsString> {
    std::env::var_os(key)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a `getenv` closure from explicit key→value pairs; any key not listed resolves to
    /// absent. Keeps each test's environment shape visible at the call site.
    fn env_of<'a>(pairs: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |key| {
            pairs
                .iter()
                .find_map(|(k, v)| (*k == key).then(|| OsString::from(*v)))
        }
    }

    /// Precedence ladder: `--socket` outranks `$SPECTER_SOCK` outranks the convention cascade, and
    /// an explicit override is `Pinned` (no fall-through) while the convention is a `Cascade`.
    #[test]
    fn resolve_precedence_cli_over_env_over_convention() {
        let env = env_of(&[("SPECTER_SOCK", "/tmp/env.sock")]);

        // `--socket` present: wins even with `$SPECTER_SOCK` set.
        assert_eq!(
            resolve(Some(Path::new("/tmp/cli.sock")), &env).unwrap(),
            Resolution::Pinned(SocketCandidate {
                source: SocketSource::Cli,
                path: PathBuf::from("/tmp/cli.sock"),
            }),
        );

        // No flag, `$SPECTER_SOCK` set: the env override pins.
        assert_eq!(
            resolve(None, &env).unwrap(),
            Resolution::Pinned(SocketCandidate {
                source: SocketSource::Env,
                path: PathBuf::from("/tmp/env.sock"),
            }),
        );

        // Neither: the convention cascade, never a pin.
        assert!(matches!(
            resolve(None, env_of(&[])).unwrap(),
            Resolution::Cascade(_),
        ));
    }

    /// A relative override (flag or env) is a hard error tagged with its source, never a silent
    /// fall-through; the message names the source and the absolute-path requirement.
    #[test]
    fn resolve_relative_override_is_hard_error() {
        let cli_err = resolve(Some(Path::new("relative/x.sock")), env_of(&[])).unwrap_err();
        assert!(matches!(
            cli_err,
            ResolveError::Relative {
                source: SocketSource::Cli,
                ..
            },
        ));
        let msg = cli_err.to_string();
        assert!(
            msg.contains("--socket") && msg.contains("absolute"),
            "got: {msg}",
        );

        let env_err = resolve(None, env_of(&[("SPECTER_SOCK", "rel/x.sock")])).unwrap_err();
        assert!(matches!(
            env_err,
            ResolveError::Relative {
                source: SocketSource::Env,
                ..
            },
        ));
    }

    /// The length check reserves the bind-time staging suffix: a path at exactly the limit
    /// resolves, one byte longer is rejected.
    #[test]
    fn resolve_overlength_override_rejected_at_staging_threshold() {
        let at_limit = format!("/{}", "a".repeat(MAX_SOCKET_PATH_LEN - 1));
        assert_eq!(at_limit.len(), MAX_SOCKET_PATH_LEN);
        assert!(
            resolve(Some(Path::new(&at_limit)), env_of(&[])).is_ok(),
            "a path at the staging threshold must resolve",
        );

        let over_limit = format!("/{}", "a".repeat(MAX_SOCKET_PATH_LEN));
        assert_eq!(over_limit.len(), MAX_SOCKET_PATH_LEN + 1);
        let over_err = resolve(Some(Path::new(&over_limit)), env_of(&[])).unwrap_err();
        assert!(
            matches!(
                over_err,
                ResolveError::TooLong {
                    source: SocketSource::Cli,
                    ..
                },
            ),
            "an over-long --socket override is a source-attributed TooLong error",
        );
        assert!(
            over_err.to_string().contains("from --socket"),
            "the overlength error must name its source: {over_err}",
        );
    }

    /// An empty `$SPECTER_SOCK` (`export SPECTER_SOCK=`) is unset, not an empty-path pin — it falls
    /// through to the convention cascade.
    #[test]
    fn resolve_empty_env_override_is_unset() {
        assert!(matches!(
            resolve(None, env_of(&[("SPECTER_SOCK", "")])).unwrap(),
            Resolution::Cascade(_),
        ));
    }

    /// The diagnostic vocab: client labels and daemon hints per source. `Cli`/`Env` exist on every
    /// platform; the convention sources are platform-gated. Also exercises [`env_os`] against an
    /// absent var.
    #[test]
    fn vocab_label_and_daemon_hint() {
        assert_eq!(SocketSource::Cli.label(), "from --socket");
        assert_eq!(SocketSource::Env.label(), "from SPECTER_SOCK");
        assert_eq!(
            SocketSource::Cli.daemon_hint(),
            DaemonHint::OperatorProvided
        );
        assert_eq!(
            SocketSource::Env.daemon_hint(),
            DaemonHint::OperatorProvided
        );
        // An operator-chosen path renders the bare creation advice and must NOT re-suggest the
        // override flag the operator just passed.
        let operator = DaemonHint::OperatorProvided.to_string();
        assert!(
            operator.contains("create the parent directory") && !operator.contains("--socket"),
            "operator-chosen path: bare creation advice, no circular flag re-suggestion: {operator}",
        );

        assert!(
            env_os("SPECTER_RESOLVE_PROBE_ABSENT_VAR").is_none(),
            "the production getenv maps an absent variable to None",
        );

        #[cfg(target_os = "linux")]
        {
            assert_eq!(SocketSource::LinuxSession.label(), "session runtime");
            assert_eq!(SocketSource::LinuxSystem.label(), "system runtime");
            assert_eq!(
                SocketSource::LinuxSession.daemon_hint(),
                DaemonHint::SessionRuntimeDir,
            );
            assert_eq!(
                SocketSource::LinuxSystem.daemon_hint(),
                DaemonHint::SystemdRuntimeDir,
            );
            // A convention hint names its provisioner AND carries the override escape, so a bind
            // failure stays actionable.
            let systemd = DaemonHint::SystemdRuntimeDir.to_string();
            assert!(
                systemd.contains("RuntimeDirectory=specter") && systemd.contains("--socket"),
                "convention hint must name its provisioner and carry the override escape: {systemd}",
            );
        }
        #[cfg(target_os = "macos")]
        {
            assert_eq!(SocketSource::MacosDefault.label(), "default location");
            assert_eq!(
                SocketSource::MacosDefault.daemon_hint(),
                DaemonHint::OperatorProvided,
            );
        }
        #[cfg(not(any(target_os = "linux", target_os = "macos")))]
        {
            assert_eq!(SocketSource::BsdSystem.label(), "system runtime");
            assert_eq!(SocketSource::BsdSystem.daemon_hint(), DaemonHint::BsdRcDir);
            let bsd = DaemonHint::BsdRcDir.to_string();
            assert!(
                bsd.contains("start_precmd") && bsd.contains("--socket"),
                "convention hint must name its provisioner and carry the override escape: {bsd}",
            );
        }
    }

    /// Linux with an absolute `$XDG_RUNTIME_DIR`: the session path heads the probe order with the
    /// system path as its fall-through tail, and the daemon's commit is that same head (the
    /// rendezvous invariant).
    #[cfg(target_os = "linux")]
    #[test]
    fn resolve_linux_session_head_with_system_tail() {
        let env = env_of(&[("XDG_RUNTIME_DIR", "/run/user/1000")]);
        let Resolution::Cascade(cascade) = resolve(None, &env).unwrap() else {
            panic!("absolute XDG_RUNTIME_DIR must resolve to a cascade");
        };

        let order: Vec<_> = cascade
            .clone()
            .into_iter()
            .map(|c| (c.source, c.path))
            .collect();
        assert_eq!(
            order,
            vec![
                (
                    SocketSource::LinuxSession,
                    PathBuf::from("/run/user/1000/specter.sock"),
                ),
                (
                    SocketSource::LinuxSystem,
                    PathBuf::from("/run/specter/specter.sock"),
                ),
            ],
        );

        // Daemon commit == client probe head, by construction.
        assert_eq!(
            resolve(None, &env).unwrap().into_commit(),
            cascade.into_iter().next().unwrap(),
        );
    }

    /// Linux without a usable `$XDG_RUNTIME_DIR` — absent, empty, relative, or an over-long join —
    /// collapses to the system path alone, no session tail.
    #[cfg(target_os = "linux")]
    #[test]
    fn resolve_linux_system_only_without_session_dir() {
        // An absolute-but-pathological runtime dir: joined with the socket filename it overflows
        // `sun_path`, so `validate` rejects the session rung and resolution falls through to the
        // system path — exactly as an absent/empty/relative dir does. Bound here so the `&str`
        // outlives the array borrow below.
        let over_long = format!("/{}", "x".repeat(SUN_MAX));
        for unusable in [
            None,
            Some(""),
            Some("relative/dir"),
            Some(over_long.as_str()),
        ] {
            let pairs: Vec<(&str, &str)> = unusable
                .map(|v| vec![("XDG_RUNTIME_DIR", v)])
                .unwrap_or_default();
            let env = env_of(&pairs);
            let Resolution::Cascade(cascade) = resolve(None, &env).unwrap() else {
                panic!("convention must resolve to a cascade for {unusable:?}");
            };
            let candidates: Vec<_> = cascade.into_iter().collect();
            assert_eq!(
                candidates.len(),
                1,
                "no session tail when XDG_RUNTIME_DIR is {unusable:?}",
            );
            assert_eq!(candidates[0].source, SocketSource::LinuxSystem);
            assert_eq!(
                candidates[0].path,
                PathBuf::from("/run/specter/specter.sock"),
            );
        }
    }

    /// macOS convention is the fixed `/tmp/specter.sock`, ignoring `$TMPDIR`; single candidate, and
    /// the commit is that head.
    #[cfg(target_os = "macos")]
    #[test]
    fn resolve_macos_convention_is_fixed_tmp() {
        let env = env_of(&[("TMPDIR", "/some/sandbox")]);
        let Resolution::Cascade(cascade) = resolve(None, &env).unwrap() else {
            panic!("macOS convention must resolve to a cascade");
        };
        let candidates: Vec<_> = cascade.clone().into_iter().collect();
        assert_eq!(candidates.len(), 1, "macOS has a single fixed location");
        assert_eq!(candidates[0].source, SocketSource::MacosDefault);
        assert_eq!(candidates[0].path, PathBuf::from("/tmp/specter.sock"));
        assert_eq!(
            resolve(None, &env).unwrap().into_commit(),
            cascade.into_iter().next().unwrap(),
        );
    }

    /// BSD convention is the fixed `/var/run/specter/specter.sock`; single candidate, commit == head.
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    #[test]
    fn resolve_bsd_convention_is_var_run() {
        let env = env_of(&[]);
        let Resolution::Cascade(cascade) = resolve(None, &env).unwrap() else {
            panic!("BSD convention must resolve to a cascade");
        };
        let candidates: Vec<_> = cascade.clone().into_iter().collect();
        assert_eq!(candidates.len(), 1, "BSD has a single fixed location");
        assert_eq!(candidates[0].source, SocketSource::BsdSystem);
        assert_eq!(
            candidates[0].path,
            PathBuf::from("/var/run/specter/specter.sock"),
        );
        assert_eq!(
            resolve(None, &env).unwrap().into_commit(),
            cascade.into_iter().next().unwrap(),
        );
    }
}
