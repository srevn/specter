//! `specter status -o human` renderer — key/value layout of the
//! daemon's [`StatusResponse`].
//!
//! Layout:
//!
//! ```text
//! specter status
//! ─────────────────────────────────────────────────────────────
//! uptime              0:02:13
//! started             2026-05-23T11:41:02Z
//! reloads             3 (last 2026-05-23T11:43:00Z via sighup)
//! subs                12 attached · 1 disabled (toml) · 2 disabled (runtime)
//! profiles            4 active
//! promoters           1 attached
//! config              /etc/specter.toml
//! socket              /run/user/1000/specter.sock
//! ```
//!
//! `--wide` is accepted but currently ignored — `status` carries
//! no extra fields today. The renderer keeps the flag in its
//! signature so the client need not branch on per-verb argument
//! shapes.
//!
//! Pure function: `(&StatusResponse, bool) -> String`. No I/O, no
//! styling — the current `status` view stays plain text.

use crate::ipc::protocol::StatusResponse;
use crate::ipc::wire::{WireReloadTrigger, WireTime};
use std::fmt::Write as _;

/// Render the status response as one operator-readable block.
///
/// `_wide` is reserved for future extensions (currently unused; the
/// `status` view fits on one screen of the default columns) — keeps
/// the signature aligned with the other renderers.
pub(crate) fn render(resp: &StatusResponse, _wide: bool) -> String {
    let mut out = String::with_capacity(512);
    let _ = writeln!(out, "specter status");
    let _ = writeln!(out, "{}", "─".repeat(61));
    let _ = writeln!(
        out,
        "{:LABEL_WIDTH$}{}",
        "uptime",
        format_uptime(resp.uptime_secs),
    );
    let _ = writeln!(out, "{:LABEL_WIDTH$}{}", "started", resp.start_wall);
    let _ = writeln!(
        out,
        "{:LABEL_WIDTH$}{}",
        "reloads",
        format_reloads(
            resp.reload_count,
            resp.last_reload_at.as_ref(),
            resp.last_reload_via,
        ),
    );
    let _ = writeln!(
        out,
        "{:LABEL_WIDTH$}{} attached · {} disabled (toml) · {} disabled (runtime)",
        "subs", resp.sub_total, resp.sub_disabled_toml, resp.sub_disabled_runtime,
    );
    let _ = writeln!(
        out,
        "{:LABEL_WIDTH$}{} active",
        "profiles", resp.profile_active,
    );
    let _ = writeln!(
        out,
        "{:LABEL_WIDTH$}{} attached",
        "promoters", resp.promoter_active,
    );
    // `Path::display()` returns a formatter adapter that streams into
    // `out` without materialising an intermediate `String`.
    let _ = writeln!(
        out,
        "{:LABEL_WIDTH$}{}",
        "config",
        resp.config_path.display(),
    );
    let _ = writeln!(
        out,
        "{:LABEL_WIDTH$}{}",
        "socket",
        resp.socket_path.display(),
    );
    out
}

/// Width of the label column. Padded to align all values vertically.
const LABEL_WIDTH: usize = 20;

/// Format `uptime_secs` as `D:HH:MM:SS` (days drop if 0). At
/// operator latency the resolution of "seconds since boot" is
/// already enough; sub-second precision would be noise.
fn format_uptime(uptime_secs: u64) -> String {
    let secs = uptime_secs % 60;
    let mins = (uptime_secs / 60) % 60;
    let hours = (uptime_secs / 3600) % 24;
    let days = uptime_secs / 86400;
    if days > 0 {
        format!("{days}d {hours:02}:{mins:02}:{secs:02}")
    } else {
        format!("{hours}:{mins:02}:{secs:02}")
    }
}

/// Render the `reloads` line: count + (optionally) the last reload
/// timestamp and trigger. A daemon that has never reloaded shows
/// just the count.
fn format_reloads(
    count: u64,
    last_at: Option<&WireTime>,
    via: Option<WireReloadTrigger>,
) -> String {
    match (last_at, via) {
        (Some(at), Some(trigger)) => {
            format!("{count} (last {at} via {})", trigger_label(trigger))
        }
        (Some(at), None) => format!("{count} (last {at})"),
        _ => count.to_string(),
    }
}

/// Operator-visible name for each reload trigger. Mirrors the
/// `snake_case` `serde(rename_all)` on [`WireReloadTrigger`] so the
/// human view matches the JSON shape.
const fn trigger_label(t: WireReloadTrigger) -> &'static str {
    match t {
        WireReloadTrigger::Sighup => "sighup",
        WireReloadTrigger::Auto => "auto",
        WireReloadTrigger::Ipc => "ipc",
    }
}

#[cfg(test)]
mod tests {
    use super::{format_reloads, format_uptime, render, trigger_label};
    use crate::ipc::protocol::StatusResponse;
    use crate::ipc::wire::{WireReloadTrigger, WireTime};
    use std::path::PathBuf;
    use std::time::{Duration, UNIX_EPOCH};

    fn fresh_status() -> StatusResponse {
        StatusResponse {
            uptime_secs: 0,
            start_wall: WireTime::from(UNIX_EPOCH),
            reload_count: 0,
            last_reload_at: None,
            last_reload_via: None,
            sub_total: 0,
            sub_disabled_toml: 0,
            sub_disabled_runtime: 0,
            profile_active: 0,
            promoter_active: 0,
            config_path: PathBuf::from("/etc/specter.toml"),
            socket_path: PathBuf::from("/tmp/specter-test.sock"),
        }
    }

    /// Smoke test: every field renders on its own line, no panic.
    /// The output starts with the header banner and includes the
    /// canonical socket-path + config-path footer.
    #[test]
    fn render_minimal_status_includes_every_label() {
        let s = render(&fresh_status(), false);
        assert!(s.starts_with("specter status\n"), "header present");
        for label in [
            "uptime",
            "started",
            "reloads",
            "subs",
            "profiles",
            "promoters",
            "config",
            "socket",
        ] {
            assert!(s.contains(label), "missing label {label:?} in:\n{s}");
        }
        assert!(s.contains("/tmp/specter-test.sock"), "socket path appears");
        assert!(s.contains("/etc/specter.toml"), "config path appears");
    }

    /// Uptime under 24h drops the days segment; ≥1 day shows it.
    /// The format is operator-friendly H:MM:SS / D HH:MM:SS, not
    /// Duration::Debug.
    #[test]
    fn format_uptime_renders_days_only_when_needed() {
        assert_eq!(format_uptime(0), "0:00:00");
        assert_eq!(format_uptime(61), "0:01:01");
        assert_eq!(format_uptime(3661), "1:01:01");
        assert_eq!(format_uptime(86_400), "1d 00:00:00");
        assert_eq!(format_uptime(86_400 + 3661), "1d 01:01:01");
    }

    /// Zero reloads renders the bare count; non-zero with attribution
    /// renders the full `(last ... via X)` parenthetical.
    #[test]
    fn format_reloads_minimal_and_full_lines() {
        assert_eq!(format_reloads(0, None, None), "0");

        let line = format_reloads(
            3,
            Some(&WireTime::from(UNIX_EPOCH + Duration::from_mins(2))),
            Some(WireReloadTrigger::Sighup),
        );
        assert!(
            line.starts_with("3 (last ") && line.ends_with(" via sighup)"),
            "got: {line}",
        );
    }

    /// `trigger_label` mirrors the `snake_case` serde rename so the
    /// human view's "via X" agrees with the JSON's `last_reload_via`.
    /// A future variant added to `WireReloadTrigger` without a
    /// matching arm here is a compile error (exhaustive `match`).
    #[test]
    fn trigger_label_matches_wire_form() {
        assert_eq!(trigger_label(WireReloadTrigger::Sighup), "sighup");
        assert_eq!(trigger_label(WireReloadTrigger::Auto), "auto");
        assert_eq!(trigger_label(WireReloadTrigger::Ipc), "ipc");
    }
}
