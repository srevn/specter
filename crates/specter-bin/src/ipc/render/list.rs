//! `specter list -o human` renderer — column-padded table.
//!
//! Default columns: `NAME STATE ANCHOR LAST_FIRED FIRES DISABLED`.
//! `--wide` adds: `PROFILE_ID SUB_ID DEDUP_SUPPRESSED SETTLE`.
//!
//! `&[Column]` of `fn` pointers is the dispatch — no trait, no
//! generics, no `Box`. Adding a column is one `Column { header,
//! render }` literal. The two column sets are `const &[Column]` so
//! the layout is data, not control flow.
//!
//! Cells render once per row into a [`Vec`] grid; widths derive from
//! that grid. A naive width-pass-then-render approach would re-allocate
//! every cell twice — operator-paced, but pointless.

use std::fmt::Write as _;

use crate::ipc::protocol::{DisabledSource, ListResponse, ListRow};
use crate::ipc::wire::WireStateLabel;

/// Render the response as one operator-readable block.
///
/// `wide = true` includes the four extra columns. Returns
/// `"no watches declared\n"` for an empty response so the operator
/// sees a definite signal, not blank output.
pub(crate) fn render(resp: &ListResponse, wide: bool) -> String {
    if resp.rows.is_empty() {
        return "no watches declared\n".to_string();
    }
    let columns = if wide { ALL_COLUMNS } else { DEFAULT_COLUMNS };

    // Render every cell once into a `cols × rows` grid. Indexed by
    // `[col][row]` so the width fold is a single linear pass per
    // column.
    let grid: Vec<Vec<String>> = columns
        .iter()
        .map(|col| resp.rows.iter().map(col.render).collect())
        .collect();

    let widths: Vec<usize> = columns
        .iter()
        .zip(&grid)
        .map(|(col, cells)| {
            cells
                .iter()
                .map(String::len)
                .max()
                .unwrap_or(0)
                .max(col.header.len())
        })
        .collect();

    let mut out = String::with_capacity(256 + 64 * resp.rows.len());
    write_header(&mut out, columns, &widths);
    write_separator(&mut out, &widths);
    for row_idx in 0..resp.rows.len() {
        write_row(&mut out, &grid, &widths, row_idx);
    }
    out
}

/// One column's header label and per-row cell renderer. `fn` pointer
/// (not `dyn Fn`) — every renderer is a static free function with no
/// captured state, so the indirection is one pointer load, no Box.
struct Column {
    header: &'static str,
    render: fn(&ListRow) -> String,
}

const DEFAULT_COLUMNS: &[Column] = &[
    Column {
        header: "NAME",
        render: col_name,
    },
    Column {
        header: "STATE",
        render: col_state,
    },
    Column {
        header: "ANCHOR",
        render: col_anchor,
    },
    Column {
        header: "LAST_FIRED",
        render: col_last_fired,
    },
    Column {
        header: "FIRES",
        render: col_fires,
    },
    Column {
        header: "DISABLED",
        render: col_disabled,
    },
];

const ALL_COLUMNS: &[Column] = &[
    Column {
        header: "NAME",
        render: col_name,
    },
    Column {
        header: "STATE",
        render: col_state,
    },
    Column {
        header: "ANCHOR",
        render: col_anchor,
    },
    Column {
        header: "LAST_FIRED",
        render: col_last_fired,
    },
    Column {
        header: "FIRES",
        render: col_fires,
    },
    Column {
        header: "DISABLED",
        render: col_disabled,
    },
    Column {
        header: "PROFILE_ID",
        render: col_profile_id,
    },
    Column {
        header: "SUB_ID",
        render: col_sub_id,
    },
    Column {
        header: "DEDUP_SUPPRESSED",
        render: col_dedup,
    },
    Column {
        header: "SETTLE",
        render: col_settle,
    },
];

/// Two-space inter-column padding on every join. The last column
/// also carries a trailing `"  "` — invisible in fixed-width
/// terminals, and keeps the header/data/separator writers symmetric.
const COL_GAP: &str = "  ";

fn write_header(out: &mut String, columns: &[Column], widths: &[usize]) {
    for (col, &w) in columns.iter().zip(widths) {
        let _ = write!(out, "{:<w$}{COL_GAP}", col.header);
    }
    out.push('\n');
}

/// One box-drawing dash per column-width cell, joined by [`COL_GAP`].
/// Each column's dash row matches the header / data widths exactly,
/// so the separator lines up under the labels regardless of cell
/// length.
fn write_separator(out: &mut String, widths: &[usize]) {
    for (i, &w) in widths.iter().enumerate() {
        if i > 0 {
            out.push_str(COL_GAP);
        }
        for _ in 0..w {
            out.push('─');
        }
    }
    out.push('\n');
}

fn write_row(out: &mut String, grid: &[Vec<String>], widths: &[usize], row_idx: usize) {
    for (col_idx, &w) in widths.iter().enumerate() {
        let _ = write!(out, "{:<w$}{COL_GAP}", grid[col_idx][row_idx]);
    }
    out.push('\n');
}

// --- Per-column renderers. `-` is the operator-visible "missing"
//     marker (mirrors `status_human`'s vocabulary).

fn col_name(row: &ListRow) -> String {
    row.name.clone()
}

fn col_state(row: &ListRow) -> String {
    row.state
        .map_or_else(|| "-".to_string(), |s| state_label_str(s).to_string())
}

fn col_anchor(row: &ListRow) -> String {
    row.anchor
        .as_ref()
        .map_or_else(|| "-".to_string(), ToString::to_string)
}

fn col_last_fired(row: &ListRow) -> String {
    row.last_fired_at
        .as_ref()
        .map_or_else(|| "-".to_string(), ToString::to_string)
}

fn col_fires(row: &ListRow) -> String {
    row.fire_count
        .map_or_else(|| "-".to_string(), |n| n.to_string())
}

fn col_disabled(row: &ListRow) -> String {
    match row.disabled {
        None => "-".to_string(),
        Some(DisabledSource::Runtime) => "runtime".to_string(),
        Some(DisabledSource::Toml) => "toml".to_string(),
    }
}

fn col_profile_id(row: &ListRow) -> String {
    row.profile
        .map_or_else(|| "-".to_string(), |id| id.0.to_string())
}

fn col_sub_id(row: &ListRow) -> String {
    row.sub
        .map_or_else(|| "-".to_string(), |id| id.0.to_string())
}

fn col_dedup(row: &ListRow) -> String {
    row.dedup_suppressed_count
        .map_or_else(|| "-".to_string(), |n| n.to_string())
}

fn col_settle(row: &ListRow) -> String {
    row.settle_ms
        .map_or_else(|| "-".to_string(), |ms| format!("{ms}ms"))
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

#[cfg(test)]
mod tests {
    use super::render;
    use crate::ipc::protocol::{DisabledSource, ListResponse, ListRow};
    use crate::ipc::wire::{WirePath, WireStateLabel};
    use std::path::Path;

    fn attached_row(name: &str) -> ListRow {
        ListRow {
            name: name.to_string(),
            state: Some(WireStateLabel::Idle),
            anchor: Some(WirePath::from(Path::new("/tmp/anchor"))),
            last_fired_at: None,
            fire_count: Some(0),
            dedup_suppressed_count: Some(0),
            settle_ms: Some(500),
            disabled: None,
            sub: Some(crate::ipc::protocol::WireId(11)),
            profile: Some(crate::ipc::protocol::WireId(22)),
            source_promoter: None,
        }
    }

    fn disabled_row(name: &str, source: DisabledSource) -> ListRow {
        ListRow {
            name: name.to_string(),
            state: None,
            anchor: None,
            last_fired_at: None,
            fire_count: None,
            dedup_suppressed_count: None,
            settle_ms: None,
            disabled: Some(source),
            sub: None,
            profile: None,
            source_promoter: None,
        }
    }

    /// Empty rows render as a definite "no watches declared" signal instead
    /// of a blank string, so the operator is not left guessing whether the
    /// request succeeded.
    #[test]
    fn list_table_renders_empty_rows_as_no_watches() {
        let resp = ListResponse { rows: vec![] };
        assert_eq!(render(&resp, false), "no watches declared\n");
        assert_eq!(
            render(&resp, true),
            "no watches declared\n",
            "wide mode also surfaces the no-watches signal",
        );
    }

    /// Default columns include NAME / STATE / ANCHOR / LAST_FIRED / FIRES
    /// / DISABLED, and NOT the four wide-only columns (PROFILE_ID / SUB_ID
    /// / DEDUP_SUPPRESSED / SETTLE).
    #[test]
    fn list_table_default_columns_excludes_wide_only() {
        let resp = ListResponse {
            rows: vec![attached_row("foo")],
        };
        let out = render(&resp, false);
        for label in ["NAME", "STATE", "ANCHOR", "LAST_FIRED", "FIRES", "DISABLED"] {
            assert!(out.contains(label), "missing column {label}: {out}");
        }
        for wide in ["PROFILE_ID", "SUB_ID", "DEDUP_SUPPRESSED", "SETTLE"] {
            assert!(
                !out.contains(wide),
                "default mode must not include wide column {wide}: {out}",
            );
        }
    }

    /// `--wide` adds the four extra columns. The default columns remain
    /// present (wide is additive, not replacing).
    #[test]
    fn list_table_wide_adds_profile_sub_dedup_settle() {
        let resp = ListResponse {
            rows: vec![attached_row("foo")],
        };
        let out = render(&resp, true);
        for label in [
            "NAME",
            "STATE",
            "ANCHOR",
            "LAST_FIRED",
            "FIRES",
            "DISABLED",
            "PROFILE_ID",
            "SUB_ID",
            "DEDUP_SUPPRESSED",
            "SETTLE",
        ] {
            assert!(out.contains(label), "missing column {label}: {out}");
        }
    }

    /// The separator row spans each column's width with the box-
    /// drawing dash. The separator is at least as wide as the
    /// corresponding header — never empty (`"─".repeat(0)` would
    /// regress this).
    #[test]
    fn list_table_separator_repeats_per_column_width() {
        let resp = ListResponse {
            rows: vec![attached_row("foo")],
        };
        let out = render(&resp, false);
        let mut lines = out.lines();
        let header = lines.next().expect("header line");
        let separator = lines.next().expect("separator line");
        let dash_count = separator.chars().filter(|c| *c == '─').count();
        // NAME (4) + STATE (5) + ANCHOR (6) + LAST_FIRED (10) +
        // FIRES (5) + DISABLED (8) = 38 — header widths bound the
        // separator from below.
        assert!(
            dash_count >= 38,
            "separator must repeat per-column-width; got {dash_count} dashes from {separator:?}",
        );
        assert!(
            !separator.is_empty(),
            "separator must not be empty (regression guard against `repeat(0)`)",
        );
        assert!(
            header.contains("NAME"),
            "header line carries column labels: {header:?}",
        );
    }

    /// Disabled-source labels render verbatim — `runtime` / `toml`
    /// match the `DisabledSource` snake_case serde rename.
    #[test]
    fn list_table_disabled_renders_source_label() {
        let resp = ListResponse {
            rows: vec![
                disabled_row("on_runtime", DisabledSource::Runtime),
                disabled_row("on_toml", DisabledSource::Toml),
            ],
        };
        let out = render(&resp, false);
        assert!(out.contains("runtime"), "missing 'runtime' label: {out}");
        assert!(out.contains("toml"), "missing 'toml' label: {out}");
    }
}
