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
`StandardErrorPath`, FreeBSD `daemon(8)`'s `-o` file) captures from stderr,
or to the explicit file when `destination = "file"`.

A command's output never reaches it — see
[Command output](#command-output).

## `[[watch]]`

One block per reaction. `name`, `path`, and `actions` are required;
everything else has a default.

| Field        | Default              | Purpose                                            |
|--------------|----------------------|----------------------------------------------------|
| `name`       | —                    | Identifies the watch in logs and IPC verbs. Unique.|
| `path`       | —                    | Absolute. Pending paths supported. A glob makes the watch dynamic — see below. |
| `actions`    | —                    | What runs when the watch settles. See below.       |
| `enabled`    | `true`               | `false` ⇒ inert at runtime; toggle without deleting.|
| `settle`     | `200ms`              | Quiet window after the last event.                 |
| `max_settle` | `1h`                 | Forced fire even if events keep arriving.          |
| `scope`      | `subtree-root`       | `subtree-root` \| `per-stable-file`.               |
| `events`     | (depends on scope)   | Event mask — see below.                            |
| `pattern`    | unset (match-all)    | Glob filter.                                       |
| `exclude`    | `[]`                 | Globs to exclude.                                  |
| `hidden`     | `false`              | Scan dotfiles.                                     |
| `recursive`  | `true`               | `false` lists one level, as `max_depth = 1`.       |
| `max_depth`  | unbounded            | Levels listed; above 1 only with `recursive`.      |

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
| `structure` | Entries created, deleted or renamed in a watched directory. |
| `content`   | File content (writes / appends).                        |
| `metadata`  | A file's attributes: permissions, owner, link count, timestamps. A size change is `content`. |

Whatever the mask, a watched file or directory reports its own deletion or
rename.

`subtree-root` defaults to `["structure", "content"]`; `per-stable-file`
adds `metadata` implicitly (the scope needs per-file FDs).

## Dynamic watches

A `path` holding a glob character — `*`, `?`, `[` or `{` — makes the watch
dynamic: it holds one watch per match of the pattern, each match a
directory or a regular file.

```toml
[[watch]]
name    = "logs"
path    = "/srv/*/logs"
actions = [{ exec = ["rotate", "${specter.path}"] }]
```

- **The pattern.** Absolute; a glob matches within one segment, so `**` is
  refused, as are `.` and `..` segments. The literal segments before the
  first glob are the prefix the watch anchors at (`/srv` above), symlinks
  resolved. A prefix that does not exist yet is waited for, as a pending
  static path is.
- **What the knobs describe.** `actions`, `settle`, `max_settle`, `scope`,
  `events` and the scan knobs (`pattern`, `exclude`, `hidden`, `recursive`,
  `max_depth`) describe each match's watch, as they would a static watch at
  that path. The walk that finds the matches is the daemon's own: it watches
  entries come and go (`structure`) and debounces at a fixed 150ms, with a
  2s ceiling.
- **What runs.** Matches present when the watch attaches — at start, at the
  reload that adds or replaces it, at `enable` — do not run its actions. A
  match that appears later runs them once it settles. A prefix the daemon
  saw absent delivers what it holds, when it appears, as matches that
  appeared: they run.
- **What is skipped.** A match that is a symlink or a special file anchors
  nothing; the first one is named (`discovery_unsupported_anchor_kind`).
- **A directory the daemon may not read** on the way to the matches is
  retried until the walk's 2s ceiling, then named (`discovery_denied`) and
  gone on without: its matches stay unknown until it can be read, and those
  already matched beneath it are kept. A directory never read is named
  once; one that was readable before is named again at each walk that has
  cause to read it — an event naming it, its parent listed after its own
  listing moved, or a walk of the whole pattern.
- **Matches come and go.** A match that vanishes leaves with its watch
  (`discovery_sub_reaped`); one replaced at its path keeps its watch, as a
  static watch keeps its own across a replace. Past 1000 matches the watch
  warns, once (`discovery_fanout_threshold`).
- **The control surface** lists each match as `<name>@<path>`. `disable`,
  `absorb` and `show` take the watch's own name and reach every match it
  holds; a match's own name is refused (`minted_sub_no_op`), or unknown to
  `show`. See [control.md](control.md).
- **An edit** of `actions` or `settle` — or of `scope` with `events` written
  out — applies in place, to every match. An edit of the pattern, a scan
  knob, `events`, `max_settle`, or `scope` with `events` omitted (the
  default mask moves with it) replaces the watch: its matches are found
  again, and, present when it attaches, do not run.

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

An `exec` may set a `timeout`, a duration such as `"5m"`. If its process
is still running when the timeout elapses, the daemon sends it SIGTERM,
then SIGKILL if it is still running 5s later, and logs a warning at
each.

### Pipes

A `pipe` action starts `N` processes together, wired stdout→stdin
(shell `a | b | c`). Each stage is a `{ exec = [...] }` table with its
own argv and optional per-stage `timeout`:

```toml
[[watch.actions]]
pipe = [
  { exec = ["find", "${specter.path}", "-name", "*.rs"] },
  { exec = ["xargs", "rustfmt", "--check"], timeout = "30s" },
]
```

- **Pipefail**: the pipe fails when any stage exits non-zero or is
  killed, and ends as its rightmost failed stage ended, as
  `set -o pipefail` reports it.
- When a stage fails, every stage before it that is still running is
  sent SIGTERM: what it writes can no longer reach the pipe's end. The
  stages after it finish on their own: the next one reads what the
  failed stage wrote to the end, which comes once the failed stage, and
  any process it started, has exited (see `exec` under
  [Command output](#command-output)).
- Pipes require **at least two stages** — `pipe = [{...}]` is rejected
  as `single-stage-pipe`; use top-level `exec` directly.
- A stage's `timeout` stops that stage; a `timeout` on the pipe itself
  is rejected.

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

### Command output

A command's stdin, stdout and stderr are `/dev/null`: Specter feeds a
command nothing and keeps nothing it prints. A command whose output
matters routes it itself — into a file the daemon's user can write:

```toml
[[watch]]
name    = "rebuild"
path    = "/srv/repo/src"
actions = [{ exec = ["sh", "-c", "exec cargo build >>/srv/repo/build.log 2>&1"] }]
```

or into the system log, tagged, through `logger`:

```toml
[[watch.actions]]
pipe = [
  { exec = ["sh", "-c", "exec cargo build 2>&1"] },
  { exec = ["logger", "-t", "rebuild"] },
]
```

When the build fails, so does the action, and `logger` still relays
every line the build wrote: a failing stage stops only the stages
before it ([Pipes](#pipes)).

In both, `exec` gives the command the shell's place: it is the process
Specter runs, so its exit status is the action's, and its `timeout`
signals the command itself.

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
