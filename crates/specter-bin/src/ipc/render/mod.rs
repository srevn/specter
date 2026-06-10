//! Human-readable rendering of operator-IPC responses.
//!
//! Every renderer is a pure writer: it appends a response's human-readable projection into a
//! caller-owned `String` and does no I/O. The caller owns the buffer — one-shot verbs (`status` /
//! `list` / `show`) hand over a fresh `String`; `specter tail` reuses one buffer across the event
//! stream, so the steady-state per-event path is allocation-free.
//!
//! Renderers paint semantic tokens via a [`style::Styler`] threaded as their trailing argument:
//! `Styler::Active` brackets tokens with ANSI, `Styler::Plain` is a byte-identical passthrough. The
//! Styler is resolved once per output stream by [`style::resolve`]; `-o json` output is never styled.
//!
//! - [`status`] — `specter status` key/value block.
//! - [`list`] — `specter list` padded table.
//! - [`show`] — `specter show <name>` detail block.
//! - [`diag`] — per-event line for `specter tail` / `specter wait`.
//! - [`style`] — the semantic color vocabulary (palette, `Styler`, gating) shared across all four.
//!
//! # Visibility
//!
//! Every export is `pub(crate)`. The client verb handlers reach `status::render` etc. directly;
//! nothing here is part of a published API surface.

use std::fmt::Display;

use self::style::{PadRight, Styler};

pub(crate) mod diag;
pub(crate) mod list;
pub(crate) mod show;
pub(crate) mod status;
pub(crate) mod style;

/// A left-aligned label cell painted with [`style::LABEL`] — the shared key-column primitive for
/// `status` and `show`, where a label is padded to a fixed width and the value follows on the same
/// line. The padding is applied to the plain text ([`PadRight`]) and the paint wraps the padded
/// result, so column alignment survives styling.
#[must_use]
pub(crate) fn label_cell(sty: Styler, label: &str, width: usize) -> impl Display + '_ {
    sty.paint(style::LABEL, PadRight(label, width))
}
