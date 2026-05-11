//! Pure resolver — turns the substitution-domain projection on
//! [`Effect`] plus one [`ExecAction`] step into [`CommandResolved`] argv
//! plus the standard `SPECTER_*` env-var set.
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
//! # The `(Effect, ExecAction)` split
//!
//! Per-Effect-stable values live on [`Effect`] (`anchor_path`,
//! `target_relative`, `sub_name`, `exclude`, `diff`, `key`, `forced`,
//! `correlation`); the per-instruction argv template lives on
//! [`ExecAction`]. The caller extracts the instruction from
//! `effect.program.instructions[cursor]`, projects out the
//! `ExecAction` payload of the `SpawnExec` variant, and passes both
//! references in. v1 always carries `cursor = 0` for the active
//! instruction set (`SpawnExec`-only); future variants
//! (`SpawnPredicate`, multi-stage `SpawnPipe`) pass each leaf
//! `ExecAction` in turn.
//!
//! # `target_path` is derived, not stored
//!
//! The Effect carries `anchor_path: Arc<Path>` and `target_relative:
//! CompactString`; the resolver derives `target_path`
//! (`${specter.path}` / `SPECTER_PATH`) by joining the two when
//! `target_relative` is non-empty, or by borrowing `anchor_path` when
//! it is. Subtree fires (always empty relative) avoid the `PathBuf`
//! allocation entirely; PerFile fires allocate exactly once per
//! resolve, at the spawn boundary where Latest-coalesce has already
//! filtered Effects that won't run.
//!
//! # `SPECTER_DIFF_PATH` slots in alphabetically
//!
//! The actuator's `spawn_effect` materialises the diff tmp file (path
//! depends on the actuator process's pid, so it can't be derived purely)
//! and passes the resulting `&Path` to [`resolve_step`] as `diff_path`.
//! The resolver inserts `SPECTER_DIFF_PATH` at its alphabetical position
//! in the env vec rather than relying on the caller to append after the
//! fact. The env-order golden test ([`tests::env_order_is_alphabetical`])
//! is then a guarantee about the bytes the spawned child sees, not just a
//! property of the resolver's standalone output.
//!
//! # Argv substitution semantics
//!
//! Each [`ArgTemplate`] in `ExecAction.argv` produces one or more argv
//! slots. The walk is single-pass with a prefix accumulator:
//!
//! - **Literals** and **single-value placeholders** (`${specter.path}`,
//!   `${specter.relative}`, `${specter.anchor}`, `${specter.watch}`,
//!   `${specter.parent}`, `${specter.time}`) append to the prefix.
//! - **Multi-value placeholders** (`${specter.created}`,
//!   `${specter.deleted}`, `${specter.modified}`,
//!   `${specter.renamed_from}`, `${specter.renamed_to}`,
//!   `${specter.excluded}`) emit one argv slot per source entry, each
//!   prefixed by the accumulated prefix; then the accumulator resets
//!   to empty. The first five source from `Diff`; `${specter.excluded}`
//!   sources from `effect.exclude`.
//! - At end-of-template: if anything was ever emitted from a multi-value,
//!   any remaining accumulator becomes a standalone trailing slot. If
//!   nothing was emitted (no multi-value found), the single-slot
//!   prefix is the one slot for this template.
//! - An [`ArgTemplate`] containing a multi-value placeholder that yields
//!   zero entries (empty diff list, `diff = None`, or empty exclude
//!   list) produces zero argv slots — there's no value to emit, and
//!   dropping the surrounding prefix is the principle-of-least-surprise
//!   (`["fmt", "${specter.created}"]` with no created entries is just
//!   `["fmt"]`).
//!
//! Two multi-value placeholders within one template (exotic; e.g.
//! `["mv ${specter.renamed_from} ${specter.renamed_to}"]` as one quoted
//! arg) expand **independently** — no parallel zip. Users wanting
//! per-pair semantics use `EffectScope::PerStableFile`.
//!
//! # Env catalog
//!
//! Every multi-value placeholder has an env-var counterpart whose value
//! is the same source list joined by `\n` (no trailing newline). Empty
//! list (or absent diff) renders as the empty string — unlike the argv
//! path, which drops the surrounding slot. Always-emit avoids `set -u`
//! surprises and lets shell scripts iterate uniformly with `while IFS=
//! read -r ...`. The mapping is:
//!
//! | Placeholder              | Env var                |
//! |--------------------------|------------------------|
//! | `${specter.created}`     | `SPECTER_CREATED`      |
//! | `${specter.deleted}`     | `SPECTER_DELETED`      |
//! | `${specter.modified}`    | `SPECTER_MODIFIED`     |
//! | `${specter.renamed_from}`| `SPECTER_RENAMED_FROM` |
//! | `${specter.renamed_to}`  | `SPECTER_RENAMED_TO`   |
//! | `${specter.excluded}`    | `SPECTER_EXCLUDED`     |
//!
//! The env-side surface carries segments only — inodes and rename pairing
//! live in the line-oriented `SPECTER_DIFF_PATH` tmp file.
//!
//! # Single-pass multi-value dispatch
//!
//! Both surfaces — argv prefix-tiling and env newline-joining — funnel
//! through [`for_each_multivalue`], which iterates a placeholder's
//! source list once and yields each value as `&str` via callback. The
//! argv consumer ([`substitute_one`]) tiles a clone of the accumulated
//! prefix per emitted value; the env consumer ([`env_multivalue`])
//! newline-joins into one owned `String`. The `Placeholder → source
//! list` mapping has one definition — argv and env can't drift.
//!
//! # Single rendering pass per resolve
//!
//! `format_now(now)` and the `${specter.parent}` string are computed
//! exactly once at the top of [`resolve_step`] and threaded through
//! both `substitute_argv` and `build_env`. The `${specter.time}` argv
//! slot and `SPECTER_TIME` env value, plus `${specter.parent}` and
//! `SPECTER_PARENT`, share the same source string by construction — no
//! risk of one surface formatting differently from the other under
//! future edits.

use crate::spawner::EnvVar;
use specter_core::{
    ArgPart, ArgTemplate, CommandResolved, DedupKey, Diff, Effect, ExecAction, Placeholder,
    ResourceKind,
};
use std::borrow::Cow;
use std::path::Path;
use std::time::SystemTime;

/// Resolve one [`ExecAction`] step against its owning [`Effect`] —
/// rendering argv plus the standard `SPECTER_*` env-var set. See module
/// docs.
///
/// `now` is sampled by the actuator's `spawn_effect` immediately before
/// `spawner.spawn` and reused for the `${specter.time}` argv slot AND
/// the `SPECTER_TIME` env value — they agree on the wall-clock instant
/// immediately before the kernel runs the user's command. Tests inject
/// a deterministic `now`; production sources `SystemTime::now()`.
///
/// `diff_path` is the absolute path of the actuator-materialised diff tmp
/// file when the Effect carries a [`Diff`] AND the file write succeeded;
/// otherwise `None`. The resolver emits `SPECTER_DIFF_PATH` in
/// alphabetical position iff this is `Some`, keeping env order total
/// across the spawn-time set the child observes.
#[must_use]
pub(crate) fn resolve_step<'a>(
    effect: &'a Effect,
    exec: &'a ExecAction,
    now: SystemTime,
    diff_path: Option<&'a Path>,
) -> (CommandResolved, Vec<EnvVar<'a>>) {
    // Materialise once, share with both surfaces.
    let target_path = compute_target_path(effect);
    let parent_str = parent_string(&target_path);
    let time_str = format_now(now);

    let argv = substitute_argv(exec, effect, &target_path, &parent_str, &time_str);
    // Argv done with `parent_str` / `time_str` — move them into the env
    // vec as `Cow::Owned` instead of cloning at the SPECTER_PARENT /
    // SPECTER_TIME push sites.
    let env = build_env(effect, &target_path, parent_str, time_str, diff_path);
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
///
/// Every branch is structurally a borrow of `anchor_path` (the path
/// itself, or its parent), so the function returns `&Path` and the
/// caller passes it straight to `Spawner::spawn` without an owning hop.
#[must_use]
pub(crate) fn compute_cwd(anchor_path: &Path, kind: ResourceKind) -> &Path {
    match kind {
        ResourceKind::File => anchor_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(anchor_path),
        ResourceKind::Dir | ResourceKind::Unknown => anchor_path,
    }
}

/// Derive `target_path` from `(anchor_path, target_relative)`.
///
/// Subtree fires set `target_relative` to empty (no per-entry segment) —
/// `target_path` is then the anchor itself, borrowed as `Cow::Borrowed`
/// to avoid a `PathBuf` allocation. PerFile fires set `target_relative`
/// to the diff entry's segment — `target_path = anchor_path.join(segment)`,
/// owned as `Cow::Owned`.
///
/// The empty-segment dispatch is invariant-preserving: it matches the
/// `DedupKey` variant by construction (Subtree ⇒ empty; PerFile ⇒
/// non-empty diff segment) without re-matching the variant here.
fn compute_target_path(effect: &Effect) -> Cow<'_, Path> {
    if effect.target_relative.is_empty() {
        Cow::Borrowed(&*effect.anchor_path)
    } else {
        Cow::Owned(effect.anchor_path.join(effect.target_relative.as_str()))
    }
}

/// Substitute placeholders into argv slots for one step.
fn substitute_argv(
    exec: &ExecAction,
    effect: &Effect,
    target_path: &Path,
    parent_str: &str,
    time_str: &str,
) -> Vec<String> {
    let template = &exec.argv;
    let mut argv = Vec::with_capacity(template.len());
    for arg in template {
        substitute_one(
            arg,
            effect,
            target_path,
            parent_str,
            time_str,
            effect.diff.as_deref(),
            &mut argv,
        );
    }
    argv
}

/// Render one [`ArgTemplate`] into zero or more argv slots, appending to
/// `out`.
fn substitute_one(
    arg: &ArgTemplate,
    effect: &Effect,
    target_path: &Path,
    parent_str: &str,
    time_str: &str,
    diff: Option<&Diff>,
    out: &mut Vec<String>,
) {
    let mut prefix = String::new();
    let mut emitted_any = false;
    for part in &arg.parts {
        match part {
            ArgPart::Literal(s) => prefix.push_str(s),
            ArgPart::Placeholder(p) => match p {
                Placeholder::Path => prefix.push_str(&target_path.to_string_lossy()),
                Placeholder::Relative => prefix.push_str(&effect.target_relative),
                Placeholder::Anchor => prefix.push_str(&effect.anchor_path.to_string_lossy()),
                Placeholder::Watch => prefix.push_str(&effect.sub_name),
                Placeholder::Parent => prefix.push_str(parent_str),
                Placeholder::Time => prefix.push_str(time_str),
                Placeholder::Created
                | Placeholder::Deleted
                | Placeholder::Modified
                | Placeholder::RenamedFrom
                | Placeholder::RenamedTo
                | Placeholder::Excluded => {
                    for_each_multivalue(*p, effect, diff, |v| {
                        let mut slot = prefix.clone();
                        slot.push_str(v);
                        out.push(slot);
                        emitted_any = true;
                    });
                    prefix.clear();
                }
            },
            ArgPart::EnvVar { .. } => {
                // PR1 carries the variant in core but the template
                // lexer doesn't yet emit it (Slice 2 of the
                // action-types expansion). Validation cannot produce
                // `EnvVar`, so the resolver never sees one in PR1.
                unreachable!(
                    "ArgPart::EnvVar unreachable until lexer ${{env.<NAME>}} support lands",
                );
            }
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
        // A multi-value placeholder consumed but yielded zero entries:
        // drop the entire ArgTemplate (zero argv slots). Empty-diff /
        // empty-exclude case.
    } else {
        out.push(prefix);
    }
}

fn has_multivalue(arg: &ArgTemplate) -> bool {
    arg.parts.iter().any(ArgPart::is_multivalue)
}

/// `target_path.parent()` rendered as a UTF-8-lossy [`String`], or empty
/// when `parent()` returns `None`. Shared between `${specter.parent}`
/// argv substitution and `SPECTER_PARENT` env emission so both surfaces
/// apply the same path semantics. The empty-string case is reachable
/// only for Subtree scope at the filesystem root (`target_path == "/"`);
/// see the table on [`Placeholder`].
fn parent_string(target_path: &Path) -> String {
    target_path
        .parent()
        .map_or_else(String::new, |p| p.to_string_lossy().into_owned())
}

/// Walk a multi-value placeholder's source list, yielding each value as
/// `&str` via `emit`. Zero allocation: every value is borrowed in-place
/// from `effect.exclude` (Excluded) or `diff` (the five diff-derived
/// variants).
///
/// The single-value arms (`Path`, `Relative`, `Anchor`, `Watch`,
/// `Parent`, `Time`) are unreachable by caller contract — argv routes
/// only the six multi-value variants here, env's [`env_multivalue`]
/// likewise. The empty fallback satisfies Rust's exhaustiveness.
///
/// Argv ([`substitute_one`]) tiles a clone of the accumulated prefix per
/// emitted value; env ([`env_multivalue`]) newline-joins into one owned
/// `String`. Both surfaces share this dispatch — argv and env can't drift
/// on the `Placeholder → source list` mapping.
fn for_each_multivalue(
    p: Placeholder,
    effect: &Effect,
    diff: Option<&Diff>,
    mut emit: impl FnMut(&str),
) {
    match p {
        Placeholder::Excluded => {
            for s in effect.exclude.iter() {
                emit(s.as_str());
            }
        }
        Placeholder::Created => {
            if let Some(d) = diff {
                for e in &d.created {
                    emit(e.segment.as_str());
                }
            }
        }
        Placeholder::Deleted => {
            if let Some(d) = diff {
                for e in &d.deleted {
                    emit(e.segment.as_str());
                }
            }
        }
        Placeholder::Modified => {
            if let Some(d) = diff {
                for e in &d.modified {
                    emit(e.segment.as_str());
                }
            }
        }
        Placeholder::RenamedFrom => {
            if let Some(d) = diff {
                for r in &d.renamed {
                    emit(r.from.segment.as_str());
                }
            }
        }
        Placeholder::RenamedTo => {
            if let Some(d) = diff {
                for r in &d.renamed {
                    emit(r.to.segment.as_str());
                }
            }
        }
        Placeholder::Path
        | Placeholder::Relative
        | Placeholder::Anchor
        | Placeholder::Watch
        | Placeholder::Parent
        | Placeholder::Time => {}
    }
}

/// Newline-joined render of a multi-value placeholder's source list, no
/// trailing newline. Empty list (or absent diff) renders as the empty
/// string, NOT a blank line — keeps list-content env values
/// (`SPECTER_EXCLUDED`, `SPECTER_CREATED`, etc.) distinguishable from "one
/// empty entry" for shell consumers reading via `while read`. The `first`
/// flag (instead of `out.is_empty()`) preserves the separator when an
/// interior entry is itself empty.
fn env_multivalue(p: Placeholder, effect: &Effect, diff: Option<&Diff>) -> String {
    let mut out = String::new();
    let mut first = true;
    for_each_multivalue(p, effect, diff, |v| {
        if !first {
            out.push('\n');
        }
        first = false;
        out.push_str(v);
    });
    out
}

/// Render `now` as RFC 3339 UTC second-precision (`2026-05-10T12:34:56Z`).
///
/// `humantime::format_rfc3339_seconds` panics on pre-epoch `SystemTime`.
/// Production `SystemTime::now()` never returns pre-epoch on a sane Unix
/// host, but a hostile clock or a test fixture can construct one — clamp
/// to the Unix epoch so the spawn path can't panic.
fn format_now(now: SystemTime) -> String {
    let now = now.max(SystemTime::UNIX_EPOCH);
    humantime::format_rfc3339_seconds(now).to_string()
}

/// Build the standard `SPECTER_*` env-var set. Keys land in alphabetical
/// order by name — pinned by [`tests::env_order_is_alphabetical`].
///
/// `SPECTER_DIFF_PATH` slots into its alphabetical position when
/// `diff_path` is `Some`; absent when `None`. The env order is total
/// across both cases — the spawned child observes alphabetical keys
/// regardless of whether a diff is present. Order is fixed for
/// golden-test stability and operator predictability (e.g., `env | sort`
/// matches positional `getenv` reads).
///
/// `target_path`, `parent_str`, `time_str` are pre-rendered in
/// [`resolve_step`] so this function and [`substitute_argv`] share the
/// same byte sequences for `${specter.path}` / `SPECTER_PATH`,
/// `${specter.parent}` / `SPECTER_PARENT`, and `${specter.time}` /
/// `SPECTER_TIME` respectively. `parent_str` and `time_str` arrive
/// by-value and move into the env vec via `Cow::Owned`.
///
/// Values are `Cow<'a, str>` so fields already living on the [`Effect`]
/// (anchor path lossy, `target_relative`, `sub_name`) and the
/// `diff_path` argument are emitted as `Cow::Borrowed`; values
/// synthesised here (`event_kind` is a literal, multi-value joins,
/// formatted correlation, `target_path` lossy, `parent_str`,
/// `time_str`) are emitted as `Cow::Owned`. Keys are always
/// `&'static str`.
fn build_env<'a>(
    effect: &'a Effect,
    target_path: &Path,
    parent_str: String,
    time_str: String,
    diff_path: Option<&'a Path>,
) -> Vec<EnvVar<'a>> {
    // `event_kind` derives from the DedupKey variant — Subtree/PerFile
    // by construction agree with the originating Sub's EffectScope, so
    // there is no second source-of-truth to consult.
    let event_kind: &'static str = match effect.key {
        DedupKey::Subtree { .. } => "dir-subtree",
        DedupKey::PerFile { .. } => "file",
    };
    let diff = effect.diff.as_deref();
    // 15 keys + optional SPECTER_DIFF_PATH. Pre-size to avoid the
    // resize churn under push.
    let cap = if diff_path.is_some() { 16 } else { 15 };
    let mut env: Vec<EnvVar<'a>> = Vec::with_capacity(cap);
    env.push(EnvVar {
        key: "SPECTER_ANCHOR",
        value: effect.anchor_path.to_string_lossy(),
    });
    env.push(EnvVar {
        key: "SPECTER_CORRELATION",
        value: Cow::Owned(effect.correlation.0.to_string()),
    });
    env.push(EnvVar {
        key: "SPECTER_CREATED",
        value: Cow::Owned(env_multivalue(Placeholder::Created, effect, diff)),
    });
    env.push(EnvVar {
        key: "SPECTER_DELETED",
        value: Cow::Owned(env_multivalue(Placeholder::Deleted, effect, diff)),
    });
    if let Some(p) = diff_path {
        env.push(EnvVar {
            key: "SPECTER_DIFF_PATH",
            value: p.to_string_lossy(),
        });
    }
    env.push(EnvVar {
        key: "SPECTER_EVENT_KIND",
        value: Cow::Borrowed(event_kind),
    });
    env.push(EnvVar {
        key: "SPECTER_EXCLUDED",
        value: Cow::Owned(env_multivalue(Placeholder::Excluded, effect, diff)),
    });
    env.push(EnvVar {
        key: "SPECTER_FORCED",
        value: Cow::Borrowed(if effect.forced { "1" } else { "0" }),
    });
    env.push(EnvVar {
        key: "SPECTER_MODIFIED",
        value: Cow::Owned(env_multivalue(Placeholder::Modified, effect, diff)),
    });
    env.push(EnvVar {
        key: "SPECTER_PARENT",
        value: Cow::Owned(parent_str),
    });
    env.push(EnvVar {
        key: "SPECTER_PATH",
        value: Cow::Owned(target_path.to_string_lossy().into_owned()),
    });
    env.push(EnvVar {
        key: "SPECTER_RELATIVE_PATH",
        value: Cow::Borrowed(effect.target_relative.as_str()),
    });
    env.push(EnvVar {
        key: "SPECTER_RENAMED_FROM",
        value: Cow::Owned(env_multivalue(Placeholder::RenamedFrom, effect, diff)),
    });
    env.push(EnvVar {
        key: "SPECTER_RENAMED_TO",
        value: Cow::Owned(env_multivalue(Placeholder::RenamedTo, effect, diff)),
    });
    env.push(EnvVar {
        key: "SPECTER_TIME",
        value: Cow::Owned(time_str),
    });
    env.push(EnvVar {
        key: "SPECTER_WATCH",
        value: Cow::Borrowed(effect.sub_name.as_str()),
    });
    env
}

#[cfg(test)]
#[path = "resolve_tests.rs"]
mod tests;
