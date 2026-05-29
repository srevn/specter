//! `specter show -o human` renderer — key/value block layout with
//! an indented `program` sub-block.
//!
//! Three response arms map to three layouts:
//!
//! - [`ShowResponse::Active`] — full key/value table plus program list.
//! - [`ShowResponse::Disabled`] — one line, `<name>: disabled (source)`.
//! - [`ShowResponse::Unknown`] — one line; operator hint.
//!
//! Mirror of [`super::status_human`]'s label alignment via the
//! [`LABEL_WIDTH`] constant — operators reading both views see the
//! same vertical anchor for the value column.

use std::fmt::Write as _;

use crate::ipc::protocol::{DisabledSource, ShowResponse, SubDetails};
use crate::ipc::wire::{WireAbsorbMode, WireEffectScope, WireStateLabel};

/// Render the response as one operator-readable block.
pub(crate) fn render(resp: &ShowResponse) -> String {
    match resp {
        ShowResponse::Active(d) => render_active(d),
        ShowResponse::Disabled { name, source } => {
            format!("{name}: disabled ({})\n", disabled_source_str(*source))
        }
        ShowResponse::Unknown { name } => {
            format!("{name}: unknown — not in config, not runtime-disabled\n")
        }
    }
}

/// Width of the label column. Padded so values align vertically;
/// mirrors [`super::status_human`]'s convention.
const LABEL_WIDTH: usize = 16;

/// Layout for `Active`:
///
/// ```text
/// foo
/// ────────────────────────────────────────
/// state           idle
/// anchor          /etc/specter
/// scope           subtree-root
/// settle          500ms
/// fires           7 (suppressed: 2, absorbed: 1)
/// last fired      2026-05-23T11:43:00Z
/// absorbing       until 2026-05-23T11:50:00Z (persist)
/// sub_id          1234
/// profile_id      4321
///
/// program (2 ops):
///   [0] exec /bin/build  ok→#1 fail→terminate
///   [1] exec /bin/notify  ok→escape fail→terminate
/// ```
fn render_active(d: &SubDetails) -> String {
    let mut out = String::with_capacity(512);
    let _ = writeln!(out, "{}", d.name);
    let underline_len = d.name.len().max(40);
    for _ in 0..underline_len {
        out.push('─');
    }
    out.push('\n');
    // `state: None` mirrors `anchor: None` / `last_fired_at: None`:
    // the projection surfaces a missing Profile lookup rather than
    // panicking the daemon. `-` is the operator-visible "missing"
    // marker shared with `list -o human`'s `col_state`.
    let _ = match d.state {
        Some(s) => writeln!(out, "{:LABEL_WIDTH$}{}", "state", state_label_str(s)),
        None => writeln!(out, "{:LABEL_WIDTH$}-", "state"),
    };
    let _ = match d.anchor.as_ref() {
        Some(p) => writeln!(out, "{:LABEL_WIDTH$}{}", "anchor", p),
        None => writeln!(out, "{:LABEL_WIDTH$}-", "anchor"),
    };
    let _ = writeln!(out, "{:LABEL_WIDTH$}{}", "scope", effect_scope_str(d.scope));
    let _ = writeln!(out, "{:LABEL_WIDTH$}{}ms", "settle", d.settle_ms);
    let _ = writeln!(
        out,
        "{:LABEL_WIDTH$}{} (suppressed: {}, absorbed: {})",
        "fires", d.fire_count, d.dedup_suppressed_count, d.absorb_count,
    );
    let _ = match d.last_fired_at.as_ref() {
        Some(t) => writeln!(out, "{:LABEL_WIDTH$}{}", "last fired", t),
        None => writeln!(out, "{:LABEL_WIDTH$}-", "last fired"),
    };
    // Only an armed, live window renders — the projection drops an
    // inert one, so a present `absorb` is always operator-meaningful.
    if let Some(w) = d.absorb.as_ref() {
        let _ = writeln!(
            out,
            "{:LABEL_WIDTH$}until {} ({})",
            "absorbing",
            w.expiry,
            absorb_mode_str(w.mode),
        );
    }
    if let Some(pid) = d.source_promoter {
        let _ = writeln!(out, "{:LABEL_WIDTH$}promoter {}", "source", pid.0);
    }
    let _ = writeln!(out, "{:LABEL_WIDTH$}{}", "sub_id", d.sub.0);
    let _ = writeln!(out, "{:LABEL_WIDTH$}{}", "profile_id", d.profile.0);
    out.push('\n');
    let _ = writeln!(out, "program ({} ops):", d.program.len());
    for line in &d.program {
        let _ = writeln!(out, "  {line}");
    }
    out
}

/// Operator-visible label for a [`WireStateLabel`]. Mirrors the
/// `snake_case` `serde(rename_all)` so the human view matches the
/// JSON. A future variant added to [`WireStateLabel`] without a
/// matching arm here is a compile error (exhaustive `match`).
const fn state_label_str(s: WireStateLabel) -> &'static str {
    match s {
        WireStateLabel::Idle => "idle",
        WireStateLabel::Pending => "pending",
        WireStateLabel::Batching => "batching",
        WireStateLabel::Verifying => "verifying",
        WireStateLabel::Draining => "draining",
        WireStateLabel::Awaiting => "awaiting",
        WireStateLabel::Rebasing => "rebasing",
        WireStateLabel::Settling => "settling",
    }
}

/// Operator-visible label for a [`WireEffectScope`]. Mirrors the
/// `snake_case` serde rename, with a hyphenated form already familiar
/// from the config TOML (`scope = "subtree-root"`).
const fn effect_scope_str(s: WireEffectScope) -> &'static str {
    match s {
        WireEffectScope::SubtreeRoot => "subtree-root",
        WireEffectScope::PerStableFile => "per-stable-file",
    }
}

/// Operator-visible label for a [`DisabledSource`].
const fn disabled_source_str(s: DisabledSource) -> &'static str {
    match s {
        DisabledSource::Runtime => "runtime",
        DisabledSource::Toml => "toml",
    }
}

/// Operator-visible mode label for the `absorbing until …` line.
/// Hyphenated to match this view's label table (`subtree-root`,
/// `per-stable-file`); `persist` is the bare form since the expiry
/// instant already sits on the same line. The `tail` renderer uses its
/// own snake_case table.
const fn absorb_mode_str(m: WireAbsorbMode) -> &'static str {
    match m {
        WireAbsorbMode::ConsumeOnFirst => "consume-on-first",
        WireAbsorbMode::PersistUntil => "persist",
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::ipc::protocol::{DisabledSource, ShowResponse, SubDetails, WireId};
    use crate::ipc::wire::{
        WireAbsorbMode, WireAbsorbWindow, WireEffectScope, WirePath, WireStateLabel, WireTime,
    };
    use std::time::UNIX_EPOCH;

    fn details(name: &str, anchor: Option<WirePath>, program: Vec<String>) -> SubDetails {
        details_full(name, anchor, program, None, 0)
    }

    /// `details` with explicit `absorb` window + `absorb_count` — the
    /// fold-surface fields the absorb-render tests exercise; the
    /// zero-arg `details` threads `None, 0` for every other test.
    fn details_full(
        name: &str,
        anchor: Option<WirePath>,
        program: Vec<String>,
        absorb: Option<WireAbsorbWindow>,
        absorb_count: u64,
    ) -> SubDetails {
        SubDetails {
            name: name.to_string(),
            sub: WireId(1),
            profile: WireId(2),
            state: Some(WireStateLabel::Idle),
            anchor,
            last_fired_at: None,
            fire_count: 0,
            dedup_suppressed_count: 0,
            absorb,
            absorb_count,
            settle_ms: 500,
            source_promoter: None,
            scope: WireEffectScope::SubtreeRoot,
            program,
        }
    }

    /// The program block renders as `program (N ops):` followed by each
    /// pre-rendered line indented by two spaces.
    #[test]
    fn show_human_active_renders_program_lines_indented() {
        let d = details(
            "foo",
            Some(WirePath::from(std::path::Path::new("/etc/specter"))),
            vec![
                "[0] exec /bin/build  ok→#1 fail→terminate".to_string(),
                "[1] exec /bin/notify  ok→escape fail→terminate".to_string(),
            ],
        );
        let out = render(&ShowResponse::Active(d));
        assert!(
            out.contains("program (2 ops):"),
            "program header missing: {out}"
        );
        assert!(
            out.contains("\n  [0] exec /bin/build"),
            "first program line not two-space indented: {out}",
        );
        assert!(
            out.contains("\n  [1] exec /bin/notify"),
            "second program line not two-space indented: {out}",
        );
    }

    /// Anchor-vanish (`None`) renders as `-` rather than an empty-string
    /// sentinel — list and show carry the same `Option<WirePath>`
    /// semantics on the wire.
    #[test]
    fn show_human_active_anchor_none_renders_dash() {
        let d = details("foo", None, vec![]);
        let out = render(&ShowResponse::Active(d));
        let anchor_line = out
            .lines()
            .find(|l| l.starts_with("anchor"))
            .expect("anchor line present");
        assert!(
            anchor_line.contains('-'),
            "anchor=None must render as '-': {anchor_line:?}",
        );
    }

    /// `state: None` renders as `-` — the operator-visible signal for
    /// the engine-invariant breach the projection surfaces gracefully
    /// instead of panicking. Mirrors `list -o human`'s `col_state`
    /// `None → "-"` arm; pinning it on `show` keeps the two verbs'
    /// vocabulary aligned.
    #[test]
    fn show_human_active_state_none_renders_dash() {
        let mut d = details("foo", None, vec![]);
        d.state = None;
        let out = render(&ShowResponse::Active(d));
        let state_line = out
            .lines()
            .find(|l| l.starts_with("state"))
            .expect("state line present");
        assert!(
            state_line.contains('-'),
            "state=None must render as '-': {state_line:?}",
        );
    }

    /// An armed `absorb` window renders an `absorbing   until <T> (mode)`
    /// line with the hyphenated mode label, and the `fires` line always
    /// carries the `absorbed: <n>` fold counter. A `None` window omits
    /// the `absorbing` line entirely (the projection already dropped an
    /// inert window, so a present `Some` is always operator-meaningful).
    #[test]
    fn show_human_active_renders_absorb_window_and_count() {
        let d = details_full(
            "foo",
            None,
            vec![],
            Some(WireAbsorbWindow {
                expiry: WireTime::from(UNIX_EPOCH),
                mode: WireAbsorbMode::ConsumeOnFirst,
            }),
            4,
        );
        let out = render(&ShowResponse::Active(d));
        let absorbing = out
            .lines()
            .find(|l| l.starts_with("absorbing"))
            .expect("absorbing line present when window is Some");
        assert!(
            absorbing.contains("until 1970-01-01T00:00:00Z"),
            "absorbing line carries the expiry: {absorbing:?}",
        );
        assert!(
            absorbing.contains("(consume-on-first)"),
            "absorbing line carries the hyphenated mode label: {absorbing:?}",
        );
        let fires = out
            .lines()
            .find(|l| l.starts_with("fires"))
            .expect("fires line present");
        assert!(
            fires.contains("absorbed: 4"),
            "fires line carries the fold counter: {fires:?}",
        );
    }

    /// `PersistUntil` renders the bare `persist` mode label (the expiry
    /// instant already sits on the same line), and a `None` window omits
    /// the `absorbing` line.
    #[test]
    fn show_human_active_persist_label_and_absent_window() {
        let with = details_full(
            "foo",
            None,
            vec![],
            Some(WireAbsorbWindow {
                expiry: WireTime::from(UNIX_EPOCH),
                mode: WireAbsorbMode::PersistUntil,
            }),
            0,
        );
        let out = render(&ShowResponse::Active(with));
        assert!(
            out.lines()
                .any(|l| l.starts_with("absorbing") && l.contains("(persist)")),
            "PersistUntil renders the bare `persist` label: {out}",
        );

        let without = details("foo", None, vec![]);
        let out = render(&ShowResponse::Active(without));
        assert!(
            !out.lines().any(|l| l.starts_with("absorbing")),
            "None window omits the absorbing line entirely: {out}",
        );
    }

    /// `Disabled` arm renders a one-liner naming the source.
    #[test]
    fn show_human_disabled_renders_source() {
        let r = ShowResponse::Disabled {
            name: "paused".into(),
            source: DisabledSource::Runtime,
        };
        let out = render(&r);
        assert_eq!(out, "paused: disabled (runtime)\n");

        let r2 = ShowResponse::Disabled {
            name: "off".into(),
            source: DisabledSource::Toml,
        };
        assert_eq!(render(&r2), "off: disabled (toml)\n");
    }

    /// `Unknown` arm renders a helpful hint that locates the resolution
    /// failure (typo vs runtime vs TOML) for the operator.
    #[test]
    fn show_human_unknown_renders_helpful_message() {
        let r = ShowResponse::Unknown {
            name: "ghost".into(),
        };
        let out = render(&r);
        assert!(out.contains("ghost"));
        assert!(
            out.contains("unknown"),
            "Unknown arm carries the 'unknown' keyword: {out}",
        );
        assert!(
            out.contains("not in config"),
            "Unknown arm tells the operator where to look: {out}",
        );
    }
}
