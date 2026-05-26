# Placeholders

`exec` argv slots reference Specter's substitution catalog through two
namespaces: `${specter.<name>}` for runtime-derived values and
`${env.<NAME>}` for operator environment variables. Anything else
`$`-shaped — bare `$NAME`, `${VAR}`, `$5`, or an unrecognised namespace
like `${capture.foo}` — passes through to the spawned process verbatim,
so shell, awk, perl, or make idioms inside an `sh -c` slot keep working.

## `${specter.*}`

Single-value placeholders render one string into the surrounding argv slot.

| Placeholder           | Meaning                                                |
|-----------------------|--------------------------------------------------------|
| `${specter.path}`     | Absolute path of the target (`per-stable-file`) or anchor (`subtree-root`) |
| `${specter.relative}` | Path relative to the watch anchor (empty for `subtree-root`) |
| `${specter.anchor}`   | Absolute path of the watch's anchor                    |
| `${specter.parent}`   | Parent of `${specter.path}` (empty only for a subtree-root anchored at `/`) |
| `${specter.watch}`    | Watch name (the `[[watch]] name` field)                |
| `${specter.time}`     | Wall-clock immediately before spawn, RFC 3339 UTC, second-precision (`2026-05-10T12:34:56Z`) |

Multi-value placeholders produce **one argv slot per value**, with any
surrounding literal prefix tiled into each slot; an empty list drops
the entire surrounding slot.

| Placeholder                | Source                                       |
|----------------------------|----------------------------------------------|
| `${specter.created}`       | New entries (anchor-relative segments)       |
| `${specter.deleted}`       | Deleted entries                              |
| `${specter.modified}`      | Entries with changed content                 |
| `${specter.renamed_from}`  | Source side of each rename                   |
| `${specter.renamed_to}`    | Target side of each rename                   |
| `${specter.excluded}`      | The watch's `exclude` patterns               |

Example — `rsync` with one `--exclude=` per pattern:

```toml
actions = [{ exec = ["rsync", "-av", "--exclude=${specter.excluded}", "${specter.anchor}/", "/backup/"] }]
# argv = ["rsync", "-av", "--exclude=*.tmp", "--exclude=cache/", "/srv/repo/", "/backup/"]
```

A literal **after** a multi-value placeholder (e.g.
`"--flag=${specter.excluded}-end"`) is not appended to each emitted
value — when at least one value emits, the trailing literal becomes its
own standalone argv slot. Place per-value suffixes inside the prefix or
use a wrapper script.

## `${env.*}`

Substitutes the operator's own environment (sampled **once at startup**,
not per spawn). The reference is **strict**: a missing variable with no
default fails the plan with a clear log line — Specter never silently
renders the empty string for a misconfigured TOML. Opt into lenient
explicitly with a `:-` default.

| Form                          | Behaviour                                                 |
|-------------------------------|-----------------------------------------------------------|
| `${env.HOME}`                 | strict: unset ⇒ plan fails                                |
| `${env.HOME:-/tmp}`           | unset ⇒ render `/tmp`                                     |
| `${env.HOME:-}`               | unset ⇒ render empty string (explicit lenient opt-in)     |

Names must match `[A-Za-z_][A-Za-z0-9_]*`; defaults are frozen literals
(no nested placeholders). The captured snapshot is immutable for the
actuator's lifetime — `SIGHUP` does **not** re-read the environment.
Use `${env.<NAME>}` only when the operator-side env genuinely doesn't
change at runtime.

## Child environment

The spawned child receives a `SPECTER_*` set in addition to the
inherited parent environment.

| Variable                | Value                                                                |
|-------------------------|----------------------------------------------------------------------|
| `SPECTER_PATH`          | mirrors `${specter.path}`                                            |
| `SPECTER_RELATIVE_PATH` | mirrors `${specter.relative}`                                        |
| `SPECTER_ANCHOR`        | mirrors `${specter.anchor}`                                          |
| `SPECTER_PARENT`        | mirrors `${specter.parent}`                                          |
| `SPECTER_WATCH`         | mirrors `${specter.watch}`                                           |
| `SPECTER_TIME`          | mirrors `${specter.time}` (same instant — resolver samples once per spawn) |
| `SPECTER_EXCLUDED`      | mirrors `${specter.excluded}` (newline-separated, no trailing newline) |
| `SPECTER_CREATED`       | mirrors `${specter.created}` (newline-separated, empty when no diff) |
| `SPECTER_DELETED`       | mirrors `${specter.deleted}` (newline-separated, empty when no diff) |
| `SPECTER_MODIFIED`      | mirrors `${specter.modified}` (newline-separated, empty when no diff) |
| `SPECTER_RENAMED_FROM`  | mirrors `${specter.renamed_from}` (newline-separated, empty when no diff) |
| `SPECTER_RENAMED_TO`    | mirrors `${specter.renamed_to}` (newline-separated, empty when no diff) |
| `SPECTER_EVENT_KIND`    | `dir-subtree` or `file`                                              |
| `SPECTER_FORCED`        | `0` or `1` — `1` when the burst crossed `max_settle` before settling |
| `SPECTER_CORRELATION`   | per-Effect monotonic decimal id                                      |
| `SPECTER_DIFF_PATH`     | tab-separated diff file (set only when the watch's actions reference diff-derived placeholders or `scope = "per-stable-file"`; shared across every step of a multi-step plan; removed once the plan exits) |

## Escape rules

- `$$` — a literal `$`. Use `$$$$` to pass `$$` through to a shell that
  wants to expand its PID.
- `${specter.<unknown>}` and `${env.<INVALID>}` — typo guards fire
  (fail-loud) only inside the explicit namespaces. `${specter.PATH}`,
  `${specter.}`, `${env.1FOO}` (digit-first), and unterminated
  `${specter.path` error during config validation.

## Pass-through

Anything that doesn't match `${specter.<name>}` or `${env.<NAME>}`
exactly — `$path`, `${specter}` (no dot), `${SPECTER.path}` (uppercase
prefix), `${HOME}` (no namespace) — is literal pass-through. The shell,
if any, sees those bytes unchanged.
