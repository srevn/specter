//! Signal-handler installation + signal-pipeline surface.
//!
//! [`register_signal_handlers`] runs at `App::run`'s prologue, *before*
//! any other production action. [`SignalPipe::new`] installs
//! `sa_sigaction` for each of [`HANDLED_SIGNALS`] on construction —
//! any signal arriving during the rest of init is captured by the
//! signal-pipeline's internal pipe (owned by the returned
//! [`SignalPipe`]) and surfaces on the first reactor tick's
//! `TOKEN_SIGNAL` drain. Without this lift, SIGTERM during config
//! load would fall through to the kernel default (immediate process
//! death) and bypass orderly shutdown.
//!
//! [`SignalPipe`] exists for one reason: expose the signal pipe's
//! read end via [`AsFd`] so [`crate::driver::hub::DriverHub`]
//! registers it through [`mio::unix::SourceFd`] uniformly with the
//! other kernel-fd Sources (watcher, config_watcher). Holding one
//! [`SignalPipe`] live for the daemon's lifetime closes the
//! registration-tear-down gap an intermediate drop-then-recreate
//! dance during boot would have opened (signal-hook unregisters our
//! handler chain entries on [`SignalPipe`] drop; a gap window would
//! leak signals to the kernel default).
//!
//! The double-tap shutdown vocabulary ([`HARD_EXIT_WINDOW`],
//! [`HARD_EXIT_CODE`], [`HARD_SHUTDOWN_CONFIRM_TIMEOUT`]) lives here so
//! the registration site and the dispatch consumer share one source of
//! truth.

use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
use signal_hook::iterator::backend::SignalDelivery;
use signal_hook::iterator::exfiltrator::SignalOnly;
use std::io;
use std::os::fd::{AsFd, BorrowedFd};
use std::os::unix::net::UnixStream;
use std::time::Duration;

/// The signal set the bin handles end-to-end: SIGHUP for reload,
/// SIGINT / SIGTERM for shutdown (with double-tap escalation). Pinned
/// in one place so the registration site ([`crate::app::run`]'s
/// prologue) and the signal thread's iterator can't drift apart.
pub(crate) const HANDLED_SIGNALS: [i32; 3] = [SIGHUP, SIGINT, SIGTERM];

/// Max gap between two terminations before the second escalates to a
/// hard exit. Operator pressing Ctrl-C twice in <2s → "I'm done waiting."
pub(crate) const HARD_EXIT_WINDOW: Duration = Duration::from_secs(2);

/// Exit code conventionally used for "killed by SIGINT" (128 + 2).
pub(crate) const HARD_EXIT_CODE: i32 = 130;

/// Upper bound on how long the signal thread waits for the actuator's
/// phase 3 confirmation pulse before calling `exit_fn` regardless.
///
/// Healthy phase 3 fanout is microseconds per child (a `kill(2)`
/// syscall); the pulse arrives well inside this window. The timeout
/// is the bound for a *wedged* actuator — a wait thread deadlocked,
/// a panic during the fanout, etc. — past which the parent must die
/// even without confirmation: the kernel reaps surviving children
/// on parent exit, and an orphan window > a few hundred milliseconds
/// is already operator-visible.
///
/// `200ms` is 4× the historical 50ms sleep heuristic, generous enough
/// for cross-thread hop + SIGKILL syscalls on a large child set under
/// scheduler contention, tight enough to keep double-Ctrl-C
/// responsive.
pub(crate) const HARD_SHUTDOWN_CONFIRM_TIMEOUT: Duration = Duration::from_millis(200);

/// Owner of the signal-pipeline's reactor-visible surface.
///
/// Built on [`SignalDelivery`] directly (not
/// `signal_hook::iterator::Signals`) because the latter's inner
/// delivery field is module-private — there is no path through it to
/// the read-end fd needed for mio [`SourceFd`](mio::unix::SourceFd)
/// registration. The `backend` module is signal-hook's documented
/// substrate for async-runtime adapter crates; using it here matches
/// that contract.
///
/// The inner pipe is a [`std::os::unix::net::UnixStream`] pair (not
/// mio's flavor): the only read site is `SignalDelivery::flush` which
/// uses `MSG_DONTWAIT` on every `recv`, syscall-level non-blocking
/// regardless of FD-level `O_NONBLOCK`. The looser FD default is
/// preferable — any future plain-read code path blocks visibly rather
/// than silently `EAGAIN`-ing.
///
/// Exists for one reason: expose the signal pipe's read end via
/// [`AsFd`] so the Hub registers it through [`SourceFd`](mio::unix::SourceFd)
/// uniformly with the other kernel-fd Sources (watcher, config_watcher).
#[derive(Debug)]
pub(crate) struct SignalPipe {
    delivery: SignalDelivery<UnixStream, SignalOnly>,
}

impl SignalPipe {
    /// Construct a fresh signal pipeline, installing `sa_sigaction` for
    /// every signal in [`HANDLED_SIGNALS`] before returning.
    ///
    /// `SignalDelivery::with_pipe`'s constructor loop calls
    /// `Handle::add_signal` per entry, which synchronously runs
    /// `signal_hook_registry::register_sigaction` before the loop
    /// iteration returns — by the time this function returns, every
    /// handler in the set is installed. The prologue-before-config-load
    /// discipline at [`crate::app::run`] depends on this synchronicity.
    pub(crate) fn new() -> io::Result<Self> {
        let (read, write) = UnixStream::pair()?;
        let delivery = SignalDelivery::with_pipe(read, write, SignalOnly, HANDLED_SIGNALS)?;
        Ok(Self { delivery })
    }

    /// Drain queued signals from the pipe.
    ///
    /// Non-blocking by construction: `SignalDelivery::pending` calls
    /// an internal `flush` that uses `MSG_DONTWAIT` on every `recv`,
    /// then walks a 128-entry exfiltrator slot array pure-cpu. Signals
    /// are returned in ascending signal-number order; duplicates within
    /// one drain window coalesce to one entry per signal.
    pub(crate) fn pending(&mut self) -> impl Iterator<Item = i32> + '_ {
        self.delivery.pending()
    }
}

impl AsFd for SignalPipe {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.delivery.get_read().as_fd()
    }
}

/// Register `sa_sigaction` handlers for [`HANDLED_SIGNALS`] and return
/// the reactor-visible [`SignalPipe`].
///
/// Called from `App::run`'s prologue — *before* config load,
/// observability init, and channel allocation. The kernel installs
/// the handlers synchronously: any signal arriving in the
/// initialisation window after this call is captured by the
/// signal-pipeline's internal pipe (owned by the returned
/// [`SignalPipe`]) and surfaces on the first reactor tick's
/// `TOKEN_SIGNAL` drain once the Hub has registered the pipe fd
/// against [`mio::Poll`]. Without this lift, every line of init ran
/// with SIGTERM's kernel-default disposition (immediate process
/// death) — see [`crate::app::run`] for the longer rationale.
///
/// The returned value is consumed by
/// [`crate::driver::hub::DriverHub::new`] which registers the pipe
/// fd as the reactor's `TOKEN_SIGNAL` source. Holding one
/// [`SignalPipe`] for the daemon's lifetime means `sa_sigaction`
/// stays installed from the moment this call returns until process
/// exit — no drop-then-recreate gap where signals could leak to the
/// kernel default.
pub(crate) fn register_signal_handlers() -> io::Result<SignalPipe> {
    SignalPipe::new()
}
