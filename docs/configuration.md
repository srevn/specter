# Configuration

Specter loads a single TOML file. Pass it via `--config` (or `-c`); CLI
overrides win over file values, file values over defaults.

```toml
[log]
level       = "info"            # trace | debug | info | warn | error
destination = "stderr"          # stderr | file
# path      = "/var/log/specter.log"   # required when destination = "file"

[[watch]]
name      = "rebuild"
path      = "/srv/repo/src"
actions   = [{ exec = ["cargo", "build"] }]
```

## `[log]`

Engine telemetry. Goes to whatever the supervisor (systemd journal, launchd
`StandardOutPath`, FreeBSD `daemon(8)`'s `-o` file) captures from stderr,
or to the explicit file when `destination = "file"`.

Subprocess output is per-watch — see `log_output` below.

## `[[watch]]`

One block per reaction. `name`, `path`, and `actions` are required;
everything else has a default.

| Field        | Default              | Purpose                                            |
|--------------|----------------------|----------------------------------------------------|
| `name`       | —                    | Identifies the watch in logs and IPC verbs. Unique.|
| `path`       | —                    | Absolute. Pending paths supported.                 |
| `actions`    | —                    | What runs when the watch settles. See below.       |
| `enabled`    | `true`               | `false` ⇒ inert at runtime; toggle without deleting.|
| `settle`     | `200ms`              | Quiet window after the last event.                 |
| `max_settle` | `1h`                 | Forced fire even if events keep arriving.          |
| `scope`      | `subtree-root`       | `subtree-root` \| `per-stable-file`.               |
| `events`     | (depends on scope)   | Event mask — see below.                            |
| `pattern`    | unset (match-all)    | Glob filter.                                       |
| `exclude`    | `[]`                 | Globs to exclude.                                  |
| `hidden`     | `false`              | Scan dotfiles.                                     |
| `recursive`  | `true`               | Descend into subdirectories.                       |
| `max_depth`  | unbounded            | Cap descent depth.                                 |
| `log_output` | `false`              | Forward child stdout/stderr to specter's stdio.    |

### `max_settle`

Default `1h`. If `actions` has multiple steps, `max_settle * 4` is the
recovery budget for the whole plan — tune up for long sequences (a
5-step plan that legitimately runs 25 min would otherwise trip the
hatch). The same budget bounds the sum of per-step `timeout`s plus the
SIGTERM→SIGKILL grace (5s by default per step); a plan whose timeouts
collectively exceed `max_settle * 4` will see the engine force a rebase
before every timer fires.

### `scope`

- `subtree-root` (default) — the engine fires **one** Effect per stable
  burst, against the anchor as a whole. `${specter.path}` is the anchor.
- `per-stable-file` — the engine fires **one Effect per stable file**
  inside the anchor. `${specter.path}` is the file. `${specter.relative}`
  is its anchor-relative segment.

### `events`

Per-event-class mask. Default depends on scope:

| Class       | Meaning                                                 |
|-------------|---------------------------------------------------------|
| `structure` | Creates, deletes, renames, attribute changes.           |
| `content`   | File content (writes / appends).                        |
| `metadata`  | mtime / size / fs_id changes without content writes.    |

`subtree-root` defaults to `["structure", "content"]`; `per-stable-file`
adds `metadata` implicitly (the scope needs per-file FDs).

## Actions

`actions` is an array. Each entry runs in sequence, **stop-on-failure**:
if step N fails, steps N+1..M don't run and the plan reports `Failed`.
The engine's outstanding-effect accounting is per-plan (one
`EffectComplete` per plan, regardless of step count).

The inline form:

```toml
actions = [
  { exec = ["cargo", "test"] },
  { exec = ["./scripts/deploy.sh"] },
]
```

The block-table form reads better for longer entries:

```toml
[[watch.actions]]
exec = ["cargo", "test"]

[[watch.actions]]
exec = ["./scripts/deploy.sh"]
```

Each `exec` slot is argv-only — no shell expansion. For shell semantics,
wrap with `sh -c`.

### Pipes

A `pipe` action wires `N` processes stdout→stdin (shell `a | b | c`).
Each stage is a `{ exec = [...] }` table with its own argv and optional
per-stage `timeout`:

```toml
[[watch.actions]]
pipe = [
  { exec = ["find", "${specter.path}", "-name", "*.rs"] },
  { exec = ["xargs", "rustfmt", "--check"], timeout = "30s" },
]
```

- Stages spawn in parallel; the kernel's SIGPIPE chain wires their I/O.
- **Pipefail-on**: any stage's non-zero exit fails the whole pipe.
  Aggregated outcome — `exit_code` is the **last** non-zero exit in
  spawn order (matches `set -o pipefail`); `signal` is the **first**
  signal observed (a per-stage timeout's SIGTERM dominates a later
  natural exit).
- A failing stage triggers SIGTERM cascade to every alive sibling, so
  a hung downstream stage can't keep the pipe alive after upstream
  failure.
- Pipes require **at least two stages** — `pipe = [{...}]` is rejected
  as `single-stage-pipe`; use top-level `exec` directly.
- Top-level `timeout` on a pipe is rejected — deadline per-stage on the
  nested `exec`.

### Conditionals

A `when` / `then` / `else` action runs a predicate first and branches
on its outcome. The predicate's outcome does **not** propagate to the
plan's terminus — `when` is a branch, not a guard.

```toml
[[watch.actions]]
when = { exec = ["cargo", "test", "--quiet"], timeout = "5m" }
then = [{ exec = ["./scripts/deploy.sh"] }]
else = [{ exec = ["notify-send", "tests failed"] }]
```

- Predicate `Ok` ⇒ run `then` in order, stop-on-failure.
- Predicate `Failed` (non-zero exit, signal, spawn failure, or predicate
  timeout) ⇒ run `else` (if present); otherwise skip past the conditional.
- `else` is optional. Without it, predicate `Failed` is a no-op and the
  plan continues with the next action.
- The predicate carries its own `timeout` inside `when`; the action
  itself does not accept a top-level `timeout`.
- `then` and `else` are full recursive action arrays — nested
  conditionals and pipes are both legal.

## Hot reload

Three triggers, same apply path:

| Trigger                   | How                                                     |
|---------------------------|---------------------------------------------------------|
| `SIGHUP`                  | `kill -HUP <pid>`, `systemctl reload specter`, etc.     |
| IPC `specter reload`      | Operator client over the UNIX socket.                   |
| Auto file-watch (default) | Edit-and-save; disable with `--no-config-watch`.        |

Auto-reload is on by default but the operator can disable it (network
filesystems, retargeted symlinks, replaced parent directories — the
watcher's preconditions fail in those scenarios). SIGHUP and IPC reload
always work.

Reload computes a diff against the current registry; only changed,
added, or removed watches touch the engine. Watches that are currently
runtime-disabled (via `specter disable`) stay disabled across reload as
long as their `[[watch]]` entry remains in the file.

## Placeholders

`exec` argv slots accept `${specter.<name>}` and `${env.<NAME>}`
substitutions; the child process also receives a `SPECTER_*` environment
set. See [placeholders.md](placeholders.md) for the full catalog and
rendering rules.
