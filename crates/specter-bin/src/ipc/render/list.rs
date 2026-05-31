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
//! Each cell renders once per row into a [`Vec`] grid as a [`Cell`] —
//! its plain text plus the [`anstyle::Style`] to paint it with. Column
//! widths derive from the plain `text.len()`, so the SGR bytes a
//! `Styler::Active` paint adds never shift a column: stripping the ANSI
//! from a styled render reproduces the plain layout byte-for-byte.

use std::fmt::Write as _;

use anstyle::Style;

use crate::ipc::protocol::{ListResponse, ListRow};
use crate::ipc::render::style::{self, PadRight, Rule, Styler};

/// Render the response as one operator-readable block into the
/// caller's buffer.
///
/// `wide = true` includes the four extra columns. Writes
/// `no watches declared\n` for an empty response so the operator sees
/// a definite signal, not blank output. `sty` gates ANSI styling on
/// the resolved stdout stream.
pub(crate) fn render(out: &mut String, resp: &ListResponse, wide: bool, sty: Styler) {
    if resp.rows.is_empty() {
        out.push_str("no watches declared\n");
        return;
    }
    let columns = if wide { ALL_COLUMNS } else { DEFAULT_COLUMNS };

    // Render every cell once into a `cols × rows` grid. Indexed by
    // `[col][row]` so the width fold is a single linear pass per
    // column.
    let grid: Vec<Vec<Cell>> = columns
        .iter()
        .map(|col| resp.rows.iter().map(col.render).collect())
        .collect();

    let widths: Vec<usize> = columns
        .iter()
        .zip(&grid)
        .map(|(col, cells)| {
            cells
                .iter()
                .map(|c| c.text.len())
                .max()
                .unwrap_or(0)
                .max(col.header.len())
        })
        .collect();

    out.reserve(256 + 64 * resp.rows.len());
    write_header(out, columns, &widths, sty);
    write_separator(out, &widths, sty);
    for row_idx in 0..resp.rows.len() {
        write_row(out, &grid, &widths, row_idx, sty);
    }
}

/// One column's header label and per-row cell renderer. `fn` pointer
/// (not `dyn Fn`) — every renderer is a static free function with no
/// captured state, so the indirection is one pointer load, no Box.
struct Column {
    header: &'static str,
    render: fn(&ListRow) -> Cell,
}

/// One rendered table cell — its plain text plus the [`Style`] to
/// paint it with. Widths fold over `text.len()` (plain bytes), so a
/// styled cell occupies the same columns as its plain twin.
///
/// The paint wraps the right-padded cell ([`write_row`]), so the style
/// covers the trailing pad: cell styles must be glyph-only (foreground
/// hue or `dimmed`) — never background / underline / reverse, which
/// would render on the pad spaces and break column alignment to the eye
/// (the same constraint [`style::LABEL`] documents).
struct Cell {
    text: String,
    style: Style,
}

impl Cell {
    /// An unstyled value cell.
    const fn data(text: String) -> Self {
        Self {
            text,
            style: Style::new(),
        }
    }

    /// The `-` placeholder for an absent value, painted
    /// [`style::MISSING`].
    fn missing() -> Self {
        Self {
            text: "-".to_string(),
            style: style::MISSING,
        }
    }

    /// A value cell painted with an explicit style.
    const fn styled(text: String, style: Style) -> Self {
        Self { text, style }
    }
}

/// Map an `Option` to a cell: `Some` through `f`, `None` to the shared
/// [`Cell::missing`] marker. Collapses the `Option → "-"` arm every
/// optional column would otherwise restate — including the styled
/// present-arms (`state`, `disabled`, the ids), since `f` returns any
/// [`Cell`].
fn present_or_missing<T>(value: Option<T>, f: impl FnOnce(T) -> Cell) -> Cell {
    value.map_or_else(Cell::missing, f)
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

fn write_header(out: &mut String, columns: &[Column], widths: &[usize], sty: Styler) {
    for (col, &w) in columns.iter().zip(widths) {
        let _ = write!(
            out,
            "{}{COL_GAP}",
            sty.paint(style::LABEL, PadRight(col.header, w))
        );
    }
    out.push('\n');
}

/// One box-drawing dash per column-width cell, joined by [`COL_GAP`].
/// Each column's dash row matches the header / data widths exactly,
/// so the separator lines up under the labels regardless of cell
/// length. The dash run is a [`Rule`] per column painted
/// [`style::DELIM`] — the N-rules-joined-by-gap shape, not one rule.
fn write_separator(out: &mut String, widths: &[usize], sty: Styler) {
    for (i, &w) in widths.iter().enumerate() {
        if i > 0 {
            out.push_str(COL_GAP);
        }
        let _ = write!(out, "{}", sty.paint(style::DELIM, Rule(w)));
    }
    out.push('\n');
}

fn write_row(out: &mut String, grid: &[Vec<Cell>], widths: &[usize], row_idx: usize, sty: Styler) {
    for (col_idx, &w) in widths.iter().enumerate() {
        let cell = &grid[col_idx][row_idx];
        let _ = write!(
            out,
            "{}{COL_GAP}",
            sty.paint(cell.style, PadRight(&cell.text, w))
        );
    }
    out.push('\n');
}

// --- Per-column renderers. Each returns a [`Cell`]; `-` (painted
//     [`style::MISSING`]) is the operator-visible "missing" marker
//     shared with `show`'s vocabulary. `state` carries its phase hue,
//     `disabled` is [`style::OFF`], the ids are [`style::SECONDARY`];
//     every other value is unstyled data.

fn col_name(row: &ListRow) -> Cell {
    Cell::data(row.name.clone())
}

fn col_state(row: &ListRow) -> Cell {
    present_or_missing(row.state, |s| Cell::styled(s.to_string(), style::state(s)))
}

fn col_anchor(row: &ListRow) -> Cell {
    present_or_missing(row.anchor.as_ref(), |p| Cell::data(p.to_string()))
}

fn col_last_fired(row: &ListRow) -> Cell {
    present_or_missing(row.last_fired_at.as_ref(), |t| Cell::data(t.to_string()))
}

fn col_fires(row: &ListRow) -> Cell {
    present_or_missing(row.fire_count, |n| Cell::data(n.to_string()))
}

fn col_disabled(row: &ListRow) -> Cell {
    present_or_missing(row.disabled, |src| {
        Cell::styled(src.to_string(), style::OFF)
    })
}

fn col_profile_id(row: &ListRow) -> Cell {
    present_or_missing(row.profile, |id| {
        Cell::styled(id.0.to_string(), style::SECONDARY)
    })
}

fn col_sub_id(row: &ListRow) -> Cell {
    present_or_missing(row.sub, |id| {
        Cell::styled(id.0.to_string(), style::SECONDARY)
    })
}

fn col_dedup(row: &ListRow) -> Cell {
    present_or_missing(row.dedup_suppressed_count, |n| Cell::data(n.to_string()))
}

fn col_settle(row: &ListRow) -> Cell {
    present_or_missing(row.settle_ms, |ms| Cell::data(format!("{ms}ms")))
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::ipc::protocol::{DisabledSource, ListResponse, ListRow};
    use crate::ipc::render::style::Styler;
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
        let mut buf = String::new();
        render(&mut buf, &resp, false, Styler::Plain);
        assert_eq!(buf, "no watches declared\n");
        let mut buf = String::new();
        render(&mut buf, &resp, true, Styler::Plain);
        assert_eq!(
            buf, "no watches declared\n",
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
        let mut out = String::new();
        render(&mut out, &resp, false, Styler::Plain);
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
        let mut out = String::new();
        render(&mut out, &resp, true, Styler::Plain);
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
        let mut out = String::new();
        render(&mut out, &resp, false, Styler::Plain);
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
        let mut out = String::new();
        render(&mut out, &resp, false, Styler::Plain);
        assert!(out.contains("runtime"), "missing 'runtime' label: {out}");
        assert!(out.contains("toml"), "missing 'toml' label: {out}");
    }

    /// Geometry parity: an `Active` render of a multi-row table,
    /// stripped of every SGR escape, is byte-identical to the `Plain`
    /// render in both default and wide modes — proving the painted cells
    /// occupy exactly the same columns (widths fold over the plain
    /// `text.len()`, never the SGR bytes a paint adds).
    #[test]
    fn list_active_preserves_column_geometry() {
        use crate::ipc::render::style::strip_ansi;

        let resp = ListResponse {
            rows: vec![
                attached_row("alpha"),
                disabled_row("bravo", DisabledSource::Runtime),
                attached_row("charlie-longer-name"),
            ],
        };
        for wide in [false, true] {
            let mut active = String::new();
            render(&mut active, &resp, wide, Styler::Active);
            let mut plain = String::new();
            render(&mut plain, &resp, wide, Styler::Plain);
            assert!(
                active.contains('\x1b'),
                "Active emits SGR (wide={wide}): {active:?}",
            );
            assert_eq!(
                strip_ansi(&active),
                plain,
                "stripped Active must equal Plain (wide={wide}) — geometry preserved",
            );
        }
    }
}
