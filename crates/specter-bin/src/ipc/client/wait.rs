//! `specter wait <name>` client handler.
//!
//! Subscribes by name (server resolves `name → SubId` atomic with
//! `add_subscriber`; an unknown name fails the Subscribe immediately
//! with `ERR_UNKNOWN_SUB`), then reads streamed [`WireDiagnostic`]s
//! until one matches the requested `--kind` (or the deadline fires,
//! or the stream ends without a match).
//!
//! # Exit codes
//!
//! - `0` — matched the requested kind. The matching event renders
//!   to stdout for the operator's confirmation (one human line).
//! - `1` — connect / subscribe failure (including the daemon's
//!   structured `ERR_UNKNOWN_SUB` error response).
//! - `2` — precondition violated mid-wait: `--kind fire` observed a
//!   [`WireDiagnostic::SubDetached`] before the requested fire. The
//!   Sub is gone; no fire is coming, an indefinite wait would hang
//!   for an event that will never arrive.
//! - `124` — deadline elapsed (POSIX `timeout(1)` convention).
//! - other non-zero ([`ExitCode::FAILURE`]) — stream ended without a
//!   match (daemon shutdown closed the conn, peer terminated mid-
//!   stream, etc.).
//!
//! # Deadline mechanics
//!
//! `--timeout` is the *total* budget from handler entry to match.
//! The deadline is captured once before subscribing (so the budget
//! covers the connect / ack handshake too) and re-applied as the
//! remaining time before every read. The connect-time 5s ack
//! deadline is intentionally kept for the ack itself: if `--timeout
//! < 5s` and the daemon hangs the ack, the user sees a "receive
//! failed: …" message (a daemon problem) instead of the `124`
//! "didn't fire in time" exit code — the two failure modes stay
//! distinguishable.
//!
//! `Duration::ZERO` on `set_read_timeout` is implementation-defined
//! on Linux (older glibc returned `EINVAL`; newer kernels treat it
//! as non-blocking). The handler explicitly maps a zero / past-due
//! remaining to exit `124` *before* the syscall, so the syscall
//! never sees a zero argument.

use compact_str::CompactString;
use specter_config::{WaitArgs, WaitKind};
use std::io::{self, Write};
use std::process::ExitCode;
use std::time::Instant;

use crate::ipc::client::subscribe;
use crate::ipc::render::diag_human;
use crate::ipc::wire::WireDiagnostic;

/// Run the `specter wait` stream loop.
pub(crate) fn run(args: &WaitArgs) -> ExitCode {
    let deadline = args.timeout.map(|d| Instant::now() + d);

    let name = CompactString::from(args.name.as_str());
    let mut sub = match subscribe::open(&args.client, "wait", Some(name)) {
        Ok(s) => s,
        Err(code) => return code,
    };

    // Indefinite wait: clear the connect-time 5s deadline once.
    // Re-clearing on every iteration would waste a syscall per
    // event with no observable effect.
    if deadline.is_none()
        && let Err(e) = sub.set_read_timeout(None)
    {
        eprintln!("specter wait: clear read deadline failed: {e}");
        return ExitCode::FAILURE;
    }

    loop {
        // Per-iteration deadline application: read horizon shrinks
        // with each iteration. The pre-syscall zero check converts
        // `now >= deadline` into a clean `124` exit instead of a
        // platform-dependent `set_read_timeout(0)` call.
        if let Some(d) = deadline {
            let Some(remaining) = d.checked_duration_since(Instant::now()) else {
                return ExitCode::from(124);
            };
            if remaining.is_zero() {
                return ExitCode::from(124);
            }
            if let Err(e) = sub.set_read_timeout(Some(remaining)) {
                eprintln!("specter wait: set deadline failed: {e}");
                return ExitCode::FAILURE;
            }
        }

        match sub.read_next() {
            Ok(Some(wire)) => match classify(args.kind, &wire) {
                Match::Matched => return emit_matched(&wire),
                Match::DetachBeforeFire => {
                    eprintln!("specter wait: target detached before fire");
                    return ExitCode::from(2);
                }
                Match::Skip => {}
            },
            Ok(None) => {
                eprintln!("specter wait: daemon disconnected before match");
                return ExitCode::FAILURE;
            }
            Err(e)
                if e.kind() == io::ErrorKind::TimedOut || e.kind() == io::ErrorKind::WouldBlock =>
            {
                return ExitCode::from(124);
            }
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                eprintln!("specter wait: malformed diagnostic line: {e}");
            }
            Err(e) => {
                eprintln!("specter wait: read failed: {e}");
                return ExitCode::FAILURE;
            }
        }
    }
}

/// Classification outcome for one streamed event against the
/// requested `--kind`.
enum Match {
    /// The event is the one the operator was waiting for. Render +
    /// exit `0`.
    Matched,
    /// `--kind fire` observed a [`WireDiagnostic::SubDetached`]. The
    /// Sub is gone; no fire is coming. Exit `2`.
    DetachBeforeFire,
    /// Not the requested kind — keep reading. Per-Sub server-side
    /// filtering guarantees we only see events naming the resolved
    /// Sub, so this arm covers in-stream events like
    /// [`WireDiagnostic::SubRebound`] (a `modified_params` rebind)
    /// or post-fire effects from earlier bursts.
    Skip,
}

/// Classify one event against the wait kind.
///
/// The per-Sub server-side filter guarantees every reachable event
/// names the resolved Sub, so this match only branches on the
/// variant tag. A new per-Sub diagnostic variant (added to both
/// [`specter_core::Diagnostic`] and `crate::driver::forward`'s
/// `diag_sub_id` projection) reaches the `Skip` arm by default;
/// that's the right behaviour — only Fire/Detach are wait-actionable.
const fn classify(kind: WaitKind, wire: &WireDiagnostic) -> Match {
    match (kind, wire) {
        (WaitKind::Fire, WireDiagnostic::SubFired { .. })
        | (WaitKind::Detach, WireDiagnostic::SubDetached { .. }) => Match::Matched,
        (WaitKind::Fire, WireDiagnostic::SubDetached { .. }) => Match::DetachBeforeFire,
        _ => Match::Skip,
    }
}

/// Render the matching event to stdout (one human line) and return
/// `ExitCode::SUCCESS`. A stdout write failure surfaces as
/// [`ExitCode::FAILURE`] — the match succeeded but the operator
/// won't see the confirmation line, which is itself a signal worth
/// preserving in the exit code.
fn emit_matched(wire: &WireDiagnostic) -> ExitCode {
    let mut stdout = io::stdout().lock();
    let rendered = diag_human::render(wire);
    if let Err(e) = stdout
        .write_all(rendered.as_bytes())
        .and_then(|()| stdout.flush())
    {
        // A BrokenPipe on the confirmation line still represents a
        // matched wait — operators piping `wait` into something that
        // closes early want the same `0` they'd get from a
        // non-piped run. Everything else is a write failure.
        if e.kind() == io::ErrorKind::BrokenPipe {
            return ExitCode::SUCCESS;
        }
        eprintln!("specter wait: write failed: {e}");
        return ExitCode::FAILURE;
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::{Match, classify};
    use crate::ipc::protocol::WireId;
    use crate::ipc::wire::{WireDetachReason, WireDiagnostic, WireTime};
    use specter_config::WaitKind;
    use std::time::UNIX_EPOCH;

    fn sub_fired() -> WireDiagnostic {
        WireDiagnostic::SubFired {
            at: WireTime::from(UNIX_EPOCH),
            sub: WireId(1),
            profile: WireId(2),
            count: 1,
        }
    }

    fn sub_detached(reason: WireDetachReason) -> WireDiagnostic {
        WireDiagnostic::SubDetached {
            at: WireTime::from(UNIX_EPOCH),
            sub: WireId(1),
            profile: WireId(2),
            reason,
        }
    }

    fn sub_rebound() -> WireDiagnostic {
        WireDiagnostic::SubRebound {
            at: WireTime::from(UNIX_EPOCH),
            sub: WireId(1),
        }
    }

    /// `--kind fire` against `SubFired` matches — the happy path.
    #[test]
    fn classify_fire_matches_subfired() {
        assert!(matches!(
            classify(WaitKind::Fire, &sub_fired()),
            Match::Matched,
        ));
    }

    /// `--kind detach` against `SubDetached` matches regardless of
    /// reason (`ConfigDiffRemoved`, `IpcDisabled`, `PromoterReaped`,
    /// `ConfigDiffIdentityChanged`). Operators waiting for "this
    /// Sub left" don't differentiate the cause.
    #[test]
    fn classify_detach_matches_subdetached_any_reason() {
        for reason in [
            WireDetachReason::ConfigDiffRemoved,
            WireDetachReason::ConfigDiffIdentityChanged,
            WireDetachReason::IpcDisabled,
            WireDetachReason::PromoterReaped,
        ] {
            assert!(matches!(
                classify(WaitKind::Detach, &sub_detached(reason)),
                Match::Matched,
            ));
        }
    }

    /// `--kind fire` observing a detach is a precondition violation —
    /// the Sub is gone, no fire is coming. The handler exits `2`.
    /// Distinct from `124` (timeout) and `1` (subscribe failure) so
    /// scripts can branch on the cause.
    #[test]
    fn classify_fire_observes_detach_returns_detach_before_fire() {
        assert!(matches!(
            classify(WaitKind::Fire, &sub_detached(WireDetachReason::IpcDisabled)),
            Match::DetachBeforeFire,
        ));
    }

    /// `--kind detach` observing a fire is normal — fires happen
    /// during a detach-wait. Skip and keep reading until the detach
    /// (or the deadline) arrives.
    #[test]
    fn classify_detach_observes_fire_skips() {
        assert!(matches!(
            classify(WaitKind::Detach, &sub_fired()),
            Match::Skip,
        ));
    }

    /// `SubRebound` (a `modified_params` rebind, per-Sub but
    /// neither a fire nor a detach) is `Skip` for both kinds — the
    /// Sub is alive and in the engine, the operator's wait should
    /// continue.
    #[test]
    fn classify_subrebound_is_skip_for_both_kinds() {
        assert!(matches!(
            classify(WaitKind::Fire, &sub_rebound()),
            Match::Skip,
        ));
        assert!(matches!(
            classify(WaitKind::Detach, &sub_rebound()),
            Match::Skip,
        ));
    }
}
