//! Semantic color vocabulary for the operator-IPC renderers.
//!
//! Three concerns live here, each a separate layer:
//!
//! - **Palette** — the [`Style`] each semantic role carries. Roles (`LABEL` / `DELIM` / …) are
//!   named consts; phase and severity hues are computed by [`state`] and [`severity_style`].
//!   Several roles fold onto one weight/hue tier today (e.g. `DELIM` / `MISSING` / `SECONDARY` are
//!   all `dimmed`); naming them by role lets one be retuned without disturbing the siblings.
//! - **Mechanism** — [`Styler`] (`Active` / `Plain`, resolved once per stream) and the [`Painted`]
//!   lazy [`Display`] adapter [`Styler::paint`] returns. Plain output is byte-identical to an
//!   unstyled render and carries no added allocation, so `specter tail` keeps its amortized buffer.
//!   The [`PaintLeaf`] sealed marker enforces the no-nesting discipline at compile time.
//! - **Gating** — [`resolve`] is the sole site touching `anstyle-query` and
//!   [`std::io::IsTerminal`]; it maps a [`ColorWhen`] + [`Stream`] to a [`Styler`].
//!
//! `specter-config`'s `cli.rs` owns a small parallel palette for clap's own `--help` output
//! (`HEADING` / `LITERAL` / `PLACEHOLDER`); that surface is clap's, not the renderers', so the two
//! intentionally do not share a module.

use std::fmt::{self, Display, Formatter, Write as _};
use std::io;

use anstyle::{AnsiColor, Style};
use specter_config::ColorWhen;

use crate::ipc::protocol::WireErrorCode;
use crate::ipc::wire::{WireStateLabel, WireTime};

// --- Palette: semantic roles → `Style`. ------------------------------

/// Keys, headers, column labels, block titles. Bold, no hue — so the trailing spaces a [`PadRight`]
/// introduces stay invisible (bold over a space reads identically to a plain space). Never give
/// `LABEL` an underline / reverse / strikethrough effect: that *would* render on the pad and break
/// column alignment to the eye.
pub(crate) const LABEL: Style = Style::new().bold();

/// Structural punctuation — the `=` between a key and its value, the
/// `·` separators on the status `subs` line, the `─` rule rows.
pub(crate) const DELIM: Style = Style::new().dimmed();

/// The `-` placeholder for an absent value (`list` cells, `show` lines). De-emphasised so populated
/// cells read first.
pub(crate) const MISSING: Style = Style::new().dimmed();

/// Secondary detail the operator rarely needs front-and-centre — the `tail` line's leading
/// timestamp, the wide-only id columns, the `show` unknown-name hint.
pub(crate) const SECONDARY: Style = Style::new().dimmed();

/// The `disabled` keyword and a disabled watch's source — a watch that is deliberately off.
pub(crate) const OFF: Style = AnsiColor::Red.on_default();

/// Operator-error text on stderr (the `specter <verb>: …` line, an unknown-name report).
pub(crate) const ERR: Style = AnsiColor::Red.on_default();

/// The closed-set error *code* inside a structured failure line — bold so it stands out from the
/// surrounding [`ERR`] amplification for the operator scripting against it.
pub(crate) const ERR_CODE: Style = AnsiColor::Red.on_default().bold();

/// Hue for an operator-display phase. `Idle` rests at the default (unstyled, so painting it is a
/// no-op); `Pending` is cyan; the pre-fire phases are yellow; the post-fire phases are blue.
#[must_use]
pub(crate) const fn state(label: WireStateLabel) -> Style {
    match label {
        WireStateLabel::Idle => Style::new(),
        WireStateLabel::Pending => AnsiColor::Cyan.on_default(),
        WireStateLabel::Batching | WireStateLabel::Verifying | WireStateLabel::Draining => {
            AnsiColor::Yellow.on_default()
        }
        WireStateLabel::Awaiting | WireStateLabel::Rebasing | WireStateLabel::Settling => {
            AnsiColor::Blue.on_default()
        }
    }
}

/// Severity tier of a streamed diagnostic — the hue its variant tag carries on a `tail` / `wait`
/// line. The classifier `super::diag::severity` maps each `WireDiagnostic` variant to one of these
/// tiers (mirroring the daemon's own tracing levels — see that function's rustdoc);
/// [`severity_style`] maps the tier to its [`Style`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Severity {
    /// A violated engine invariant the daemon flags at `error!` — a malformed attach request, an
    /// anchor-kind mismatch, a walker-contract breach. The narrow red tier: genuinely wrong.
    Error,
    /// A degraded-but-recovered edge the daemon logs at `warn!` — a failed / vanished probe (the
    /// errno self-recovers), a purged claim, a forced ceiling, an overflow reseed, a gate deadline
    /// — plus the wire-only `Missed` data-loss marker. An edge was hit and handled.
    Warn,
    /// The daemon did its primary job — a watch fired. The lone green tier, elevated from the
    /// event's `info!` log level.
    Ok,
    /// Routine lifecycle, benign races, and class / consumer drops — the daemon's `info!` /
    /// `debug!` / `trace!` events. Unstyled.
    Info,
}

/// [`Style`] for a [`Severity`] tier. `Info` rests at the default (unstyled).
#[must_use]
pub(crate) const fn severity_style(severity: Severity) -> Style {
    match severity {
        Severity::Error => AnsiColor::Red.on_default(),
        Severity::Warn => AnsiColor::Yellow.on_default(),
        Severity::Ok => AnsiColor::Green.on_default().bold(),
        Severity::Info => Style::new(),
    }
}

// --- Mechanism: Styler + Painted + the PaintLeaf discipline. ---------

/// Whether a resolved output stream should carry ANSI styling. `Copy`, threaded by value into every
/// renderer; produced once per stream by [`resolve`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Styler {
    Active,
    Plain,
}

impl Styler {
    /// Wrap one leaf token in `style`. The returned [`Painted`] is a lazy [`Display`] adapter: when
    /// this `Styler` is [`Self::Active`] *and* `style` is non-empty it brackets `body` with the SGR
    /// intro / reset; otherwise it forwards `body`'s own [`Display`] verbatim — zero added
    /// allocation, byte-identical to an unstyled render.
    ///
    /// `body: PaintLeaf` is the no-nesting discipline. An ANSI reset is global, so `paint(_,
    /// paint(_, x))` would let the inner reset terminate the outer style early. [`Painted`]
    /// deliberately does not implement [`PaintLeaf`], so a nested call fails to compile. Paint
    /// leaves; concatenate siblings.
    #[must_use]
    pub(crate) fn paint<D: PaintLeaf + Display>(self, style: Style, body: D) -> Painted<D> {
        // The `style != empty` arm is a defensive optimisation, not the parity proof: `anstyle`
        // already elides both the (empty) intro and the reset for an empty style, so the `else`
        // delegate below owns Plain-path byte-equality on its own.
        let active = matches!(self, Self::Active) && style != Style::new();
        Painted {
            style,
            body,
            active,
        }
    }
}

/// Lazy [`Display`] adapter returned by [`Styler::paint`]. Formats to `body` alone unless painting
/// is active; see [`Styler::paint`].
pub(crate) struct Painted<D> {
    style: Style,
    body: D,
    active: bool,
}

impl<D: Display> Display for Painted<D> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        if self.active {
            write!(
                f,
                "{}{}{}",
                self.style.render(),
                self.body,
                self.style.render_reset(),
            )
        } else {
            // Zero-alloc passthrough — `tail` keeps its amortized buffer.
            self.body.fmt(f)
        }
    }
}

mod sealed {
    /// Sealing supertrait for [`super::PaintLeaf`]. A private module ⇒ the whitelist in the parent
    /// is the single site any leaf can be admitted; no other module (and no other crate) can widen
    /// it, and [`super::Painted`] is structurally excluded.
    pub trait Sealed {}
}

/// Marker for the leaf token types [`Styler::paint`] accepts. The impls below are the entire
/// whitelist — adding a paintable type is one `leaf!` entry. Deliberately *not* implemented for
/// [`Painted`], which is what blocks `paint(_, paint(_, x))` at compile time.
pub(crate) trait PaintLeaf: sealed::Sealed {}

// A shared reference is a leaf iff its referent is — covers `&str`, `&String`, the `&WireTime` an
// `at_field` borrow yields, etc. without a per-reference entry. `&Painted<_>` is excluded because
// `Painted` is.
impl<T: PaintLeaf + ?Sized> sealed::Sealed for &T {}
impl<T: PaintLeaf + ?Sized> PaintLeaf for &T {}

macro_rules! leaf {
    ($($t:ty),+ $(,)?) => {$(
        impl sealed::Sealed for $t {}
        impl PaintLeaf for $t {}
    )+};
}

// The exact set of types passed as `body` to `paint` across the renderers and the stderr helpers.
// Three are reached only as shared borrows via the reference blanket above — never by value: `str`
// (`&str` literals + keys), `String` (`show`'s unknown-name `&String`), and `WireTime` (an
// `at_field` borrow). The rest are painted by value (`fmt::Arguments` from the stderr helpers,
// `PadRight` / `Rule` from the layout primitives, the two wire enums painted directly).
leaf!(
    str,
    String,
    fmt::Arguments<'_>,
    PadRight<'_>,
    Rule,
    WireStateLabel,
    WireTime,
    WireErrorCode,
);

// --- Paintable layout primitives. ------------------------------------

/// Left-aligned padding to a fixed column width, computed on the **plain** text so a surrounding
/// [`Styler::paint`] wraps the padded result (SGR bytes never count toward the width). Reproduces
/// the `{:<width$}` the renderers used before color, byte-for-byte — including the pre-existing
/// byte-vs-display-width skew on multi-byte names, which this neither fixes nor worsens.
#[derive(Clone, Copy, Debug)]
pub(crate) struct PadRight<'a>(pub(crate) &'a str, pub(crate) usize);

impl Display for PadRight<'_> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{:<width$}", self.0, width = self.1)
    }
}

/// A horizontal rule — `n` box-drawing dashes (`─`) written straight
/// into the formatter, no intermediate `String::repeat` allocation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Rule(pub(crate) usize);

impl Display for Rule {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        for _ in 0..self.0 {
            f.write_char('─')?;
        }
        Ok(())
    }
}

// --- Gating: ColorWhen + Stream → Styler. ----------------------------

/// Which standard stream a [`Styler`] is being resolved for. Stdout and stderr gate independently —
/// a piped stdout with a TTY stderr still colors error lines.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Stream {
    Stdout,
    Stderr,
}

/// Resolve the effective [`Styler`] for `stream` under the operator's `--color` choice.
///
/// `Always` / `Never` are unconditional — an explicit flag overrides the environment. `Auto`
/// consults the environment and the stream's TTY state via [`auto`].
#[must_use]
pub(crate) fn resolve(when: ColorWhen, stream: Stream) -> Styler {
    let on = match when {
        ColorWhen::Always => true,
        ColorWhen::Never => false,
        ColorWhen::Auto => auto(is_tty(stream)),
    };
    if on { Styler::Active } else { Styler::Plain }
}

fn is_tty(stream: Stream) -> bool {
    use std::io::IsTerminal as _;
    match stream {
        Stream::Stdout => io::stdout().is_terminal(),
        Stream::Stderr => io::stderr().is_terminal(),
    }
}

/// `--color=auto` decision for one stream: read the three environment signals `anstyle-query` owns,
/// then apply [`auto_precedence`]. The env-VALUE semantics are `anstyle-query`'s; the ordering is
/// ours and lives in the pure core below so it stays a testable unit without touching process env
/// or a real TTY.
fn auto(is_tty: bool) -> bool {
    auto_precedence(
        anstyle_query::no_color(),
        anstyle_query::clicolor_force(),
        anstyle_query::clicolor(),
        is_tty,
    )
}

/// Precedence for `--color=auto`: `NO_COLOR` > `CLICOLOR_FORCE` > `CLICOLOR` > the stream's TTY
/// default. Pure over its four inputs (the env signals + TTY state), so the ordering is a unit test
/// rather than an end-to-end smoke run; [`auto`] supplies the live env reads.
const fn auto_precedence(
    no_color: bool,
    clicolor_force: bool,
    clicolor: Option<bool>,
    is_tty: bool,
) -> bool {
    if no_color {
        return false;
    }
    if clicolor_force {
        return true;
    }
    match clicolor {
        Some(on) => on && is_tty,
        None => is_tty,
    }
}

/// Strip ANSI SGR sequences (`ESC [ … m`) so a styled render can be compared against its plain
/// twin. Test-only, shared across the renderer test modules: stripping an `Active` render must
/// reproduce the `Plain` bytes exactly (color is purely additive). No test-dep.
#[cfg(test)]
pub(crate) fn strip_ansi(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            for c2 in chars.by_ref() {
                if c2 == 'm' {
                    break;
                }
            }
        } else {
            out.push(c);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::{
        ColorWhen, PadRight, Rule, Severity, Stream, Style, Styler, WireStateLabel,
        auto_precedence, resolve, severity_style, state, strip_ansi,
    };

    /// `Always` / `Never` are unconditional regardless of stream; `resolve` collapses them to
    /// `Active` / `Plain` without consulting the environment. (`Auto` is environment/TTY-dependent,
    /// so it is not asserted here — it is exercised end-to-end by the manual smoke run.)
    #[test]
    fn resolve_always_active_never_plain() {
        assert_eq!(resolve(ColorWhen::Always, Stream::Stdout), Styler::Active);
        assert_eq!(resolve(ColorWhen::Always, Stream::Stderr), Styler::Active);
        assert_eq!(resolve(ColorWhen::Never, Stream::Stdout), Styler::Plain);
        assert_eq!(resolve(ColorWhen::Never, Stream::Stderr), Styler::Plain);
    }

    /// `auto_precedence` ordering: `NO_COLOR` beats everything, then `CLICOLOR_FORCE`, then
    /// `CLICOLOR` (gated on the TTY), then the bare TTY default. Pure over its four inputs, so the
    /// `Auto` policy `resolve` defers to is pinned without touching process env or a real terminal
    /// (the env reads live in `auto`).
    #[test]
    fn auto_precedence_orders_no_color_force_clicolor_tty() {
        // NO_COLOR wins outright — even on a TTY with CLICOLOR_FORCE.
        assert!(!auto_precedence(true, true, Some(true), true));
        // CLICOLOR_FORCE wins over a non-TTY and CLICOLOR=off.
        assert!(auto_precedence(false, true, Some(false), false));
        // CLICOLOR=on colours only on a TTY.
        assert!(auto_precedence(false, false, Some(true), true));
        assert!(!auto_precedence(false, false, Some(true), false));
        // CLICOLOR=off suppresses even on a TTY.
        assert!(!auto_precedence(false, false, Some(false), true));
        // No env signal ⇒ the bare TTY default.
        assert!(auto_precedence(false, false, None, true));
        assert!(!auto_precedence(false, false, None, false));
    }

    /// `Styler::Plain` is a byte-identical passthrough — the painted adapter formats to the body
    /// alone, leaving plain output unchanged across every renderer.
    #[test]
    fn plain_paint_is_passthrough() {
        assert_eq!(
            Styler::Plain.paint(super::LABEL, "label").to_string(),
            "label"
        );
        assert_eq!(Styler::Plain.paint(super::ERR, "boom").to_string(), "boom");
        assert_eq!(
            Styler::Plain.paint(super::DELIM, Rule(3)).to_string(),
            "───"
        );
    }

    /// `Styler::Active` brackets a non-empty style with SGR codes; stripping them recovers the body
    /// exactly.
    #[test]
    fn active_paint_brackets_and_strips_back() {
        let painted = Styler::Active.paint(super::ERR, "boom").to_string();
        assert!(
            painted.contains('\x1b'),
            "active paint emits SGR: {painted:?}"
        );
        assert!(painted.starts_with('\x1b'), "intro leads: {painted:?}");
        assert!(painted.ends_with('m'), "reset trails: {painted:?}");
        assert_eq!(strip_ansi(&painted), "boom", "stripping recovers the body");
    }

    /// An empty [`Style`] is a passthrough even when the `Styler` is `Active` — the `paint` guard
    /// skips the (would-be empty) SGR brackets, so a `DATA`/`Info`-tier token never gains stray
    /// escape bytes.
    #[test]
    fn active_paint_empty_style_is_passthrough() {
        assert_eq!(
            Styler::Active.paint(Style::new(), "value").to_string(),
            "value",
        );
    }

    /// [`PadRight`] reproduces `{:<width$}`; [`Rule`] repeats the dash.
    #[test]
    fn primitives_match_their_format_shapes() {
        assert_eq!(PadRight("ab", 5).to_string(), "ab   ");
        assert_eq!(
            PadRight("toolong", 3).to_string(),
            "toolong",
            "no truncation"
        );
        assert_eq!(Rule(0).to_string(), "");
        assert_eq!(Rule(4).to_string(), "────");
    }

    /// Phase hues: `Idle` is the unstyled default (painting it is a no-op); every other phase
    /// carries a non-empty style.
    #[test]
    fn state_idle_is_default_others_hued() {
        assert_eq!(state(WireStateLabel::Idle), Style::new());
        for label in [
            WireStateLabel::Pending,
            WireStateLabel::Batching,
            WireStateLabel::Verifying,
            WireStateLabel::Draining,
            WireStateLabel::Awaiting,
            WireStateLabel::Rebasing,
            WireStateLabel::Settling,
        ] {
            assert_ne!(state(label), Style::new(), "{label:?} should carry a hue");
        }
    }

    /// Severity tiers: `Info` is the unstyled default; `Error` / `Warn` / `Ok` carry distinct
    /// non-empty styles.
    #[test]
    fn severity_info_is_default_others_distinct() {
        assert_eq!(severity_style(Severity::Info), Style::new());
        let error = severity_style(Severity::Error);
        let warn = severity_style(Severity::Warn);
        let ok = severity_style(Severity::Ok);
        for s in [error, warn, ok] {
            assert_ne!(s, Style::new());
        }
        assert_ne!(error, warn);
        assert_ne!(error, ok);
        assert_ne!(warn, ok);
    }
}
