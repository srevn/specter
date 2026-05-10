//! Pure resolver — turns the substitution-domain projection on
//! [`Effect`] into [`CommandResolved`] argv plus the standard `SPECTER_*`
//! env-var set.
//!
//! Lives next to the actuator because resolution runs immediately before
//! `spawner.spawn` — Latest-coalesce drops `pending` Effects before they
//! reach the syscall, so resolving at submit time wastes the bytes;
//! resolving at spawn time preserves "render late" as the architectural
//! invariant.
//!
//! Pure data work — no `std::env`, no `std::process`, no I/O. The
//! actuator's `spawn_effect` is the sole production caller; sibling unit
//! tests drive directly.
//!
//! # `SPECTER_DIFF_PATH` is **not** set here
//!
//! The actuator at spawn time materialises the diff tmp file (its path
//! depends on the actuator process's pid) and appends `SPECTER_DIFF_PATH`
//! to the env it passes to `Command::envs`. The resolver returns env
//! without that var; the actuator decides whether the file write
//! succeeded and conditionally extends env.
//!
//! # Argv substitution semantics
//!
//! Each [`ArgTemplate`] in `Effect.command.argv` produces one or more
//! argv slots. The walk is single-pass with a prefix accumulator:
//!
//! - **Literals** and **single-value placeholders** (`$path`,
//!   `$relative`, `$anchor`) append to the prefix.
//! - **Multi-value placeholders** (`$created`, `$deleted`, `$modified`,
//!   `$renamed_from`, `$renamed_to`) emit one argv slot per matching
//!   `Diff` entry, each prefixed by the accumulated prefix; then the
//!   accumulator resets to empty.
//! - At end-of-template: if anything was ever emitted from a multi-value,
//!   any remaining accumulator becomes a standalone trailing slot. If
//!   nothing was emitted (no multi-value found), the single-slot
//!   prefix is the one slot for this template.
//! - An [`ArgTemplate`] containing a multi-value placeholder that yields
//!   zero entries (empty diff list, or `diff = None`) produces zero argv
//!   slots — there's no value to emit, and dropping the surrounding
//!   prefix is the principle-of-least-surprise (`["fmt", "$created"]`
//!   with no created entries is just `["fmt"]`).
//!
//! Two multi-value placeholders within one template (exotic; e.g.
//! `["mv $renamed_from $renamed_to"]` as one quoted arg) expand
//! **independently** — no parallel zip. Users wanting per-pair semantics
//! use `EffectScope::PerStableFile`.
//!
//! # Env emission order
//!
//! Keys land in **alphabetical order** by name, a v1 break from the
//! prior logical-grouping order. Children consuming env via
//! `getenv("SPECTER_X")` are unaffected; out-of-tree consumers indexing
//! positionally would break. None observed.

use specter_core::{
    ArgPart, ArgTemplate, CommandResolved, Diff, Effect, EffectScope, Placeholder, ResourceKind,
};
use std::path::{Path, PathBuf};

/// Resolve an Effect's command + env from its substitution-domain
/// projection. See module docs.
#[must_use]
pub(crate) fn resolve_effect(effect: &Effect) -> (CommandResolved, Vec<(String, String)>) {
    let argv = substitute_argv(&effect.command.argv, effect, effect.diff.as_deref());
    let env = build_env(effect);
    (CommandResolved { argv }, env)
}

/// Choose the spawn cwd for `effect`.
///
/// `Command::current_dir` requires a directory; spawn fails with `ENOTDIR`
/// otherwise. For File-anchored Profiles the parent directory is the
/// natural cwd (user scripts use `$SPECTER_PATH` to locate the file).
/// `Dir` and `Unknown` (rare; pending paths) anchor at the path itself —
/// for `Unknown`, this may not exist on disk; the actuator surfaces such
/// failures as `EffectOutcome::Failed`.
#[must_use]
pub(crate) fn compute_cwd(anchor_path: &Path, kind: ResourceKind) -> PathBuf {
    match kind {
        ResourceKind::File => anchor_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| anchor_path.to_path_buf(), Path::to_path_buf),
        ResourceKind::Dir | ResourceKind::Unknown => anchor_path.to_path_buf(),
    }
}

/// Substitute placeholders into argv slots.
fn substitute_argv(template: &[ArgTemplate], effect: &Effect, diff: Option<&Diff>) -> Vec<String> {
    let mut argv = Vec::with_capacity(template.len());
    for arg in template {
        substitute_one(arg, effect, diff, &mut argv);
    }
    argv
}

/// Render one [`ArgTemplate`] into zero or more argv slots, appending to
/// `out`.
fn substitute_one(arg: &ArgTemplate, effect: &Effect, diff: Option<&Diff>, out: &mut Vec<String>) {
    let mut prefix = String::new();
    let mut emitted_any = false;
    for part in &arg.parts {
        match part {
            ArgPart::Literal(s) => prefix.push_str(s),
            ArgPart::Placeholder(p) => match p {
                Placeholder::Path => prefix.push_str(&effect.target_path.to_string_lossy()),
                Placeholder::Relative => prefix.push_str(&effect.target_relative),
                Placeholder::Anchor => prefix.push_str(&effect.anchor_path.to_string_lossy()),
                Placeholder::Created
                | Placeholder::Deleted
                | Placeholder::Modified
                | Placeholder::RenamedFrom
                | Placeholder::RenamedTo => {
                    let values = multivalue_values(*p, diff);
                    for v in values {
                        let mut slot = prefix.clone();
                        slot.push_str(&v);
                        out.push(slot);
                        emitted_any = true;
                    }
                    prefix.clear();
                }
            },
        }
    }
    // If a multi-value placeholder emitted at least one slot, a non-empty
    // trailing prefix becomes its own standalone slot. Otherwise the
    // prefix is the single slot for this ArgTemplate.
    if emitted_any {
        if !prefix.is_empty() {
            out.push(prefix);
        }
    } else if has_multivalue(arg) {
        // No multi-value placeholders consumed (or all yielded zero
        // entries with no leading literals/single-values). If we saw
        // *any* multi-value at all (even with zero values), the
        // ArgTemplate produces zero slots — `emitted_any == false &&
        // arg has a multi-value` is the empty-diff case.
        // Drop the prefix; zero-slot output.
    } else {
        out.push(prefix);
    }
}

fn has_multivalue(arg: &ArgTemplate) -> bool {
    arg.parts.iter().any(ArgPart::is_diff_placeholder)
}

/// Resolve a multi-value placeholder against `diff` to its segment list.
/// Returns an empty Vec if `diff` is None or the corresponding list is
/// empty.
fn multivalue_values(p: Placeholder, diff: Option<&Diff>) -> Vec<String> {
    let Some(d) = diff else {
        return Vec::new();
    };
    match p {
        Placeholder::Created => d.created.iter().map(|e| e.segment.to_string()).collect(),
        Placeholder::Deleted => d.deleted.iter().map(|e| e.segment.to_string()).collect(),
        Placeholder::Modified => d.modified.iter().map(|e| e.segment.to_string()).collect(),
        Placeholder::RenamedFrom => d
            .renamed
            .iter()
            .map(|r| r.from.segment.to_string())
            .collect(),
        Placeholder::RenamedTo => d.renamed.iter().map(|r| r.to.segment.to_string()).collect(),
        // Unreachable: caller already filtered to multi-value variants.
        Placeholder::Path | Placeholder::Relative | Placeholder::Anchor => Vec::new(),
    }
}

/// Build the standard `SPECTER_*` env-var set excluding `SPECTER_DIFF_PATH`
/// (the actuator appends that at spawn time). Keys land in alphabetical
/// order by name — pinned by [`tests::env_order_is_alphabetical`].
fn build_env(effect: &Effect) -> Vec<(String, String)> {
    let event_kind = match effect.scope {
        EffectScope::SubtreeRoot => "dir-subtree",
        EffectScope::PerStableFile => "file",
    };
    vec![
        (
            "SPECTER_ANCHOR".to_owned(),
            effect.anchor_path.to_string_lossy().into_owned(),
        ),
        (
            "SPECTER_CORRELATION".to_owned(),
            effect.correlation.0.to_string(),
        ),
        ("SPECTER_EVENT_KIND".to_owned(), event_kind.to_owned()),
        (
            "SPECTER_FORCED".to_owned(),
            if effect.forced { "1" } else { "0" }.to_owned(),
        ),
        (
            "SPECTER_PATH".to_owned(),
            effect.target_path.to_string_lossy().into_owned(),
        ),
        (
            "SPECTER_RELATIVE_PATH".to_owned(),
            effect.target_relative.to_string(),
        ),
        ("SPECTER_WATCH".to_owned(), effect.sub_name.to_string()),
    ]
}

#[cfg(test)]
#[path = "resolve_tests.rs"]
mod tests;
