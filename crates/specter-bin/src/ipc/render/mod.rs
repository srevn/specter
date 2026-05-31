//! Human-readable rendering of operator-IPC responses.
//!
//! Every renderer is a pure writer: it appends a response's
//! human-readable projection into a caller-owned `String` and does no
//! I/O. The caller owns the buffer — one-shot verbs (`status` / `list`
//! / `show`) hand over a fresh `String`; `specter tail` reuses one
//! buffer across the event stream, so the steady-state per-event path
//! is allocation-free. All output is currently monochrome.
//!
//! - [`status`] — `specter status` key/value block.
//! - [`list`] — `specter list` padded table.
//! - [`show`] — `specter show <name>` detail block.
//! - [`diag`] — per-event line for `specter tail` / `specter wait`.
//!
//! # Visibility
//!
//! Every export is `pub(crate)`. The client verb handlers reach
//! `status::render` etc. directly; nothing here is part of a
//! published API surface.

pub(crate) mod diag;
pub(crate) mod list;
pub(crate) mod show;
pub(crate) mod status;
