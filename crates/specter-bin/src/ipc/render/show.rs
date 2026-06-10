//! `specter show -o human` renderer — key/value block layout with an indented `program` sub-block.
//!
//! Three response arms map to three layouts:
//!
//! - [`ShowResponse::Active`] — full key/value table plus program list.
//! - [`ShowResponse::Disabled`] — one line, `<name>: disabled (source)`.
//! - [`ShowResponse::Unknown`] — one line; operator hint.
//!
//! Labels align to [`LABEL_WIDTH`] via the shared [`super::label_cell`] primitive, so operators
//! reading `status` and `show` see the same vertical anchor for the value column. Labels paint
//! [`style::LABEL`]; the `state` value carries its phase hue; the `Disabled` keyword is
//! [`style::OFF`] and the `Unknown` arm is [`style::ERR`]. Under `Styler::Plain` the output is
//! byte-identical to the pre-color view.

use std::fmt::Write as _;

use crate::ipc::protocol::{ShowResponse, SubDetails};
use crate::ipc::render::label_cell;
use crate::ipc::render::style::{self, Rule, Styler};
use crate::ipc::wire::{WireAbsorbMode, WireEffectScope};

/// Render the response as one operator-readable block into the caller's buffer. `sty` gates ANSI
/// styling on the resolved stdout stream.
pub(crate) fn render(out: &mut String, resp: &ShowResponse, sty: Styler) {
    match resp {
        ShowResponse::Active(d) => render_active(out, d, sty),
        ShowResponse::Disabled { name, source } => {
            let _ = writeln!(
                out,
                "{name}: {} ({source})",
                sty.paint(style::OFF, "disabled")
            );
        }
        ShowResponse::Unknown { name } => {
            let _ = writeln!(
                out,
                "{}: {} {}",
                sty.paint(style::ERR, name),
                sty.paint(style::ERR, "unknown"),
                sty.paint(style::SECONDARY, "— not in config, not runtime-disabled"),
            );
        }
    }
}

/// Width of the label column. Padded so values align vertically; mirrors [`super::status`]'s
/// convention.
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
fn render_active(out: &mut String, d: &SubDetails, sty: Styler) {
    out.reserve(512);
    let _ = writeln!(out, "{}", sty.paint(style::LABEL, d.name.as_str()));
    let _ = writeln!(
        out,
        "{}",
        sty.paint(style::DELIM, Rule(d.name.len().max(40)))
    );
    // `state: None` mirrors `anchor: None` / `last_fired_at: None`: the projection surfaces a
    // missing Profile lookup rather than panicking the daemon. `-` (painted [`style::MISSING`]) is
    // the operator-visible "missing" marker shared with `list`'s `col_state`.
    match d.state {
        Some(s) => {
            let _ = writeln!(
                out,
                "{}{}",
                label_cell(sty, "state", LABEL_WIDTH),
                sty.paint(style::state(s), s),
            );
        }
        None => missing_line(out, sty, "state"),
    }
    match d.anchor.as_ref() {
        Some(p) => {
            let _ = writeln!(out, "{}{}", label_cell(sty, "anchor", LABEL_WIDTH), p);
        }
        None => missing_line(out, sty, "anchor"),
    }
    let _ = writeln!(
        out,
        "{}{}",
        label_cell(sty, "scope", LABEL_WIDTH),
        effect_scope_str(d.scope),
    );
    let _ = writeln!(
        out,
        "{}{}ms",
        label_cell(sty, "settle", LABEL_WIDTH),
        d.settle_ms,
    );
    let _ = writeln!(
        out,
        "{}{} (suppressed: {}, absorbed: {})",
        label_cell(sty, "fires", LABEL_WIDTH),
        d.fire_count,
        d.dedup_suppressed_count,
        d.absorb_count,
    );
    match d.last_fired_at.as_ref() {
        Some(t) => {
            let _ = writeln!(out, "{}{}", label_cell(sty, "last fired", LABEL_WIDTH), t);
        }
        None => missing_line(out, sty, "last fired"),
    }
    // Only an armed, live window renders — the projection drops an inert one, so a present `absorb`
    // is always operator-meaningful.
    if let Some(w) = d.absorb.as_ref() {
        let _ = writeln!(
            out,
            "{}until {} ({})",
            label_cell(sty, "absorbing", LABEL_WIDTH),
            w.expiry,
            absorb_mode_str(w.mode),
        );
    }
    if let Some(src) = d.source_discovery {
        let _ = writeln!(
            out,
            "{}discovery {}",
            label_cell(sty, "source", LABEL_WIDTH),
            src.0,
        );
    }
    let _ = writeln!(out, "{}{}", label_cell(sty, "sub_id", LABEL_WIDTH), d.sub.0);
    let _ = writeln!(
        out,
        "{}{}",
        label_cell(sty, "profile_id", LABEL_WIDTH),
        d.profile.0,
    );
    out.push('\n');
    // The `program (N ops):` header is a section label, painted `LABEL` like the others; the lines
    // below it stay plain — the daemon pre-renders each as an opaque string the renderer does not
    // re-tokenize.
    let _ = writeln!(
        out,
        "{}",
        sty.paint(
            style::LABEL,
            format_args!("program ({} ops):", d.program.len())
        ),
    );
    for line in &d.program {
        let _ = writeln!(out, "  {line}");
    }
}

/// Write a `label   -` line for a value the projection surfaced as
/// `None` — the `-` painted [`style::MISSING`]. Shared by the three optional `Active` fields
/// (`state` / `anchor` / `last fired`).
fn missing_line(out: &mut String, sty: Styler, label: &str) {
    let _ = writeln!(
        out,
        "{}{}",
        label_cell(sty, label, LABEL_WIDTH),
        sty.paint(style::MISSING, "-"),
    );
}

/// View-local label for a [`WireEffectScope`] — hyphenated form already familiar from the config
/// TOML (`scope = "subtree-root"`). The wire's own snake-case projection lives on
/// [`WireEffectScope::as_str`]; `show.rs` chooses to diverge for the detail block.
const fn effect_scope_str(s: WireEffectScope) -> &'static str {
    match s {
        WireEffectScope::SubtreeRoot => "subtree-root",
        WireEffectScope::PerStableFile => "per-stable-file",
    }
}

/// View-local mode label for the `absorbing until …` line. Hyphenated to match this view's label
/// table (`subtree-root`, `per-stable-file`); `persist` is the bare form since the expiry instant
/// already sits on the same line. The wire's own snake-case projection lives on
/// [`WireAbsorbMode::as_str`]; `diag`/`tail` reach it through `Display`.
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
    use crate::ipc::render::style::Styler;
    use crate::ipc::wire::{
        WireAbsorbMode, WireAbsorbWindow, WireEffectScope, WirePath, WireStateLabel, WireTime,
    };
    use std::time::UNIX_EPOCH;

    fn details(name: &str, anchor: Option<WirePath>, program: Vec<String>) -> SubDetails {
        details_full(name, anchor, program, None, 0)
    }

    /// `details` with explicit `absorb` window + `absorb_count` — the fold-surface fields the
    /// absorb-render tests exercise; the zero-arg `details` threads `None, 0` for every other test.
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
            source_discovery: None,
            scope: WireEffectScope::SubtreeRoot,
            program,
        }
    }

    /// The program block renders as `program (N ops):` followed by each pre-rendered line indented
    /// by two spaces.
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
        let mut out = String::new();
        render(&mut out, &ShowResponse::Active(d), Styler::Plain);
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

    /// Anchor-vanish (`None`) renders as `-` rather than an empty-string sentinel — list and show
    /// carry the same `Option<WirePath>` semantics on the wire.
    #[test]
    fn show_human_active_anchor_none_renders_dash() {
        let d = details("foo", None, vec![]);
        let mut out = String::new();
        render(&mut out, &ShowResponse::Active(d), Styler::Plain);
        let anchor_line = out
            .lines()
            .find(|l| l.starts_with("anchor"))
            .expect("anchor line present");
        assert!(
            anchor_line.contains('-'),
            "anchor=None must render as '-': {anchor_line:?}",
        );
    }

    /// `state: None` renders as `-` — the operator-visible signal for the engine-invariant breach
    /// the projection surfaces gracefully instead of panicking. Mirrors `list -o human`'s
    /// `col_state` `None → "-"` arm; pinning it on `show` keeps the two verbs' vocabulary aligned.
    #[test]
    fn show_human_active_state_none_renders_dash() {
        let mut d = details("foo", None, vec![]);
        d.state = None;
        let mut out = String::new();
        render(&mut out, &ShowResponse::Active(d), Styler::Plain);
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
    /// line with the hyphenated mode label, and the `fires` line always carries the `absorbed: <n>`
    /// fold counter. A `None` window omits the `absorbing` line entirely (the projection already
    /// dropped an inert window, so a present `Some` is always operator-meaningful).
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
        let mut out = String::new();
        render(&mut out, &ShowResponse::Active(d), Styler::Plain);
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

    /// `PersistUntil` renders the bare `persist` mode label (the expiry instant already sits on the
    /// same line), and a `None` window omits the `absorbing` line.
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
        let mut out = String::new();
        render(&mut out, &ShowResponse::Active(with), Styler::Plain);
        assert!(
            out.lines()
                .any(|l| l.starts_with("absorbing") && l.contains("(persist)")),
            "PersistUntil renders the bare `persist` label: {out}",
        );

        let without = details("foo", None, vec![]);
        let mut out = String::new();
        render(&mut out, &ShowResponse::Active(without), Styler::Plain);
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
        let mut out = String::new();
        render(&mut out, &r, Styler::Plain);
        assert_eq!(out, "paused: disabled (runtime)\n");

        let r2 = ShowResponse::Disabled {
            name: "off".into(),
            source: DisabledSource::Toml,
        };
        let mut buf = String::new();
        render(&mut buf, &r2, Styler::Plain);
        assert_eq!(buf, "off: disabled (toml)\n");
    }

    /// `Unknown` arm renders a helpful hint that locates the resolution failure (typo vs runtime vs
    /// TOML) for the operator.
    #[test]
    fn show_human_unknown_renders_helpful_message() {
        let r = ShowResponse::Unknown {
            name: "ghost".into(),
        };
        let mut out = String::new();
        render(&mut out, &r, Styler::Plain);
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

    /// Color is purely additive across all three `show` arms: an `Active` render stripped of every
    /// SGR escape equals the `Plain` render byte-for-byte, and each arm does carry escapes (the
    /// Active table, the `disabled` keyword, the `unknown` report).
    #[test]
    fn show_active_strips_to_plain_across_arms() {
        use crate::ipc::render::style::strip_ansi;

        let active_arm = ShowResponse::Active(details(
            "foo",
            Some(WirePath::from(std::path::Path::new("/etc/specter"))),
            vec!["[0] exec /bin/build  ok→#1 fail→terminate".to_string()],
        ));
        let disabled_arm = ShowResponse::Disabled {
            name: "paused".into(),
            source: DisabledSource::Runtime,
        };
        let unknown_arm = ShowResponse::Unknown {
            name: "ghost".into(),
        };
        for resp in [&active_arm, &disabled_arm, &unknown_arm] {
            let mut active = String::new();
            render(&mut active, resp, Styler::Active);
            let mut plain = String::new();
            render(&mut plain, resp, Styler::Plain);
            assert!(active.contains('\x1b'), "Active emits SGR: {active:?}");
            assert_eq!(strip_ansi(&active), plain, "stripping Active yields Plain");
        }
    }
}
