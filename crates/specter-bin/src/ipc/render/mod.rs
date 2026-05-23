//! Human-readable rendering of operator-IPC responses.
//!
//! Every renderer is a pure `&Response → String` function. All output
//! is currently monochrome.
//!
//! - [`status_human`] — `specter status` key/value block.
//! - [`list_table`] — `specter list` padded table.
//! - [`show_human`] — `specter show <name>` detail block.
//! - [`diag_human`] — per-event line for `specter tail` / `specter wait`.
//!
//! # Visibility
//!
//! Every export is `pub(crate)`. The client verb handlers reach
//! `status_human::render` etc. directly; nothing here is part of a
//! published API surface.

pub(crate) mod diag_human;
pub(crate) mod list_table;
pub(crate) mod show_human;
pub(crate) mod status_human;
