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
//! Pure writer: `(&mut String, &StatusResponse, bool, Styler)`. No
//! I/O. The title and labels paint [`style::LABEL`], the rule and the
//! `·` separators [`style::DELIM`]; values stay unstyled. Under
//! `Styler::Plain` the output is byte-identical to the pre-color view.

use crate::ipc::protocol::{StatusResponse, WireLastReload};
use crate::ipc::render::label_cell;
use crate::ipc::render::style::{self, Rule, Styler};
use std::fmt::Write as _;

/// Render the status response as one operator-readable block.
///
/// `_wide` is reserved for future extensions (currently unused; the
/// `status` view fits on one screen of the default columns) — keeps
/// the signature aligned with the other renderers. `sty` gates ANSI
/// styling on the resolved stdout stream.
pub(crate) fn render(out: &mut String, resp: &StatusResponse, _wide: bool, sty: Styler) {
    out.reserve(512);
    let _ = writeln!(out, "{}", sty.paint(style::LABEL, "specter status"));
    let _ = writeln!(out, "{}", sty.paint(style::DELIM, Rule(61)));
    let _ = writeln!(
        out,
        "{}{}",
        label_cell(sty, "uptime", LABEL_WIDTH),
        format_uptime(resp.uptime_secs),
    );
    let _ = writeln!(
        out,
        "{}{}",
        label_cell(sty, "started", LABEL_WIDTH),
        resp.start_wall,
    );
    let _ = writeln!(
        out,
        "{}{}",
        label_cell(sty, "reloads", LABEL_WIDTH),
        format_reloads(resp.reload_count, resp.last_reload.as_ref()),
    );
    // The `·` separators paint as siblings — counts and trailing text
    // stay unstyled, so the line reads identically when plain.
    let _ = writeln!(
        out,
        "{}{} attached {} {} disabled (toml) {} {} disabled (runtime)",
        label_cell(sty, "subs", LABEL_WIDTH),
        resp.sub_total,
        sty.paint(style::DELIM, "·"),
        resp.sub_disabled_toml,
        sty.paint(style::DELIM, "·"),
        resp.sub_disabled_runtime,
    );
    let _ = writeln!(
        out,
        "{}{} active",
        label_cell(sty, "profiles", LABEL_WIDTH),
        resp.profile_active,
    );
    let _ = writeln!(
        out,
        "{}{} attached",
        label_cell(sty, "promoters", LABEL_WIDTH),
        resp.promoter_active,
    );
    // `WirePath: Display` writes its inner UTF-8 / lossy-projected
    // string verbatim — zero-alloc into `out`, no intermediate.
    let _ = writeln!(
        out,
        "{}{}",
        label_cell(sty, "config", LABEL_WIDTH),
        resp.config_path,
    );
    let _ = writeln!(
        out,
        "{}{}",
        label_cell(sty, "socket", LABEL_WIDTH),
        resp.socket_path,
    );
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

/// Render the `reloads` line: count + (optionally) the most-recent
/// reload pair. A daemon that has never reloaded shows just the
/// count. The lift to a single [`WireLastReload`] collapses the
/// prior `(Some(at), None)` defensive arm — the impossible product
/// is no longer constructable, so the match shrinks to two arms.
fn format_reloads(count: u64, last: Option<&WireLastReload>) -> String {
    match last {
        Some(r) => format!("{count} (last {} via {})", r.at, r.via),
        None => count.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::{format_reloads, format_uptime, render};
    use crate::ipc::protocol::{StatusResponse, WireLastReload};
    use crate::ipc::render::style::Styler;
    use crate::ipc::wire::{WirePath, WireReloadTrigger, WireTime};
    use std::path::Path;
    use std::time::{Duration, UNIX_EPOCH};

    fn fresh_status() -> StatusResponse {
        StatusResponse {
            uptime_secs: 0,
            start_wall: WireTime::from(UNIX_EPOCH),
            reload_count: 0,
            last_reload: None,
            sub_total: 0,
            sub_disabled_toml: 0,
            sub_disabled_runtime: 0,
            profile_active: 0,
            promoter_active: 0,
            config_path: WirePath::from(Path::new("/etc/specter.toml")),
            socket_path: WirePath::from(Path::new("/tmp/specter-test.sock")),
        }
    }

    /// Smoke test: every field renders on its own line, no panic.
    /// The output starts with the header banner and includes the
    /// canonical socket-path + config-path footer.
    #[test]
    fn render_minimal_status_includes_every_label() {
        let mut s = String::new();
        render(&mut s, &fresh_status(), false, Styler::Plain);
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
    /// renders the full `(last ... via X)` parenthetical. The
    /// impossible-by-construction `(Some(at), None)` arm collapses
    /// out of the renderer since [`WireLastReload`] holds the pair
    /// together.
    #[test]
    fn format_reloads_minimal_and_full_lines() {
        assert_eq!(format_reloads(0, None), "0");

        let lr = WireLastReload {
            at: WireTime::from(UNIX_EPOCH + Duration::from_mins(2)),
            via: WireReloadTrigger::Sighup,
        };
        let line = format_reloads(3, Some(&lr));
        assert!(
            line.starts_with("3 (last ") && line.ends_with(" via sighup)"),
            "got: {line}",
        );
    }

    /// Color is purely additive: an `Active` render stripped of every
    /// SGR escape reproduces the `Plain` render byte-for-byte, and the
    /// `Active` render does carry escapes (title / labels / rule are
    /// painted).
    #[test]
    fn status_active_strips_to_plain() {
        use crate::ipc::render::style::strip_ansi;

        let resp = fresh_status();
        let mut active = String::new();
        render(&mut active, &resp, false, Styler::Active);
        let mut plain = String::new();
        render(&mut plain, &resp, false, Styler::Plain);
        assert!(
            active.contains('\x1b'),
            "Active emits SGR escapes: {active:?}",
        );
        assert_eq!(strip_ansi(&active), plain, "stripping Active yields Plain");
    }
}
