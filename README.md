# Specter

> Prove the absence of change.

Specter watches paths, debounces bursts of events, and fires commands
when the tree settles — not when "something happened," but when
**nothing has happened for long enough** that the tree is observably
stable. Built around three pure actors:

- **Engine** — a deterministic step machine. Owns the path tree,
  per-Profile state machines, and the timer heap. Pure: no I/O, no
  threads, no `HashMap`.
- **Sensor** — kqueue watcher (BSD / macOS) plus a worker pool that
  performs directory walks. Linux/inotify is a planned port; the
  factory seam is in place.
- **Actuator** — subprocess pool. Spawns commands, coalesces by
  `DedupKey`, reaps children, reports completions.

A single bin (`specter`) wires them with bounded channels, signal
handling, hot config reload, and the `EngineDriver::tick` loop.

**Status:** alpha — single-user, no backwards-compat guarantees yet.
Tested on macOS and FreeBSD.

## Why Specter?

Conventional file-watch tools fire on every kernel event. The result is
a flurry of redundant runs against partially-written files. Specter
inverts the contract: events restart a settle timer; reactions fire
**only after the burst has decayed**, against a snapshot of the tree
that includes every change up to that quiescent point.

Concretely:

- Coarse file-tree settling — no double-fires on `git checkout`,
  multi-file editor saves, or build outputs writing dozens of
  artifacts.
- Hierarchical content hashing — re-running the same edit (saving
  with no changes, touching mtime, idempotent reformatters) does not
  re-fire the command.
- Self-event absorption — the reaction itself usually writes inside
  the watched tree; Specter folds those events into the post-fire
  rebase rather than treating them as a fresh burst.
- A built-in `--config` reload pipeline (SIGHUP) and supervisor
  templates for systemd / launchd / FreeBSD `daemon(8)`.

## Build & install

```sh
make build               # cargo build --release
make install             # → $(BINDIR)/specter (default /usr/local/bin)

# Service template (one of):
make install-systemd     # → /etc/systemd/system/specter.service
make install-launchd     # → $(LAUNCHD_DIR)/io.specter.plist
make install-freebsd     # → $(PREFIX)/etc/rc.d/specter

# All-in-one — auto-detect host OS and pick the right service template:
make install-all
```

On macOS, install-launchd / install-config default to **user scope** — the
plist lands in `~/Library/LaunchAgents/`, the config in `~/.config/specter/`,
and `launchctl bootstrap`/`bootout` use `gui/<uid>` so no `sudo` is needed.
Override `LAUNCHD_DIR=/Library/LaunchDaemons LAUNCHD_DOMAIN=system
SYSCONFDIR=/usr/local/etc` (with `sudo`) for a system install.

Linux/FreeBSD remain system-scope.

Standard variables — `PREFIX`, `DESTDIR`, `BINDIR`, `SYSCONFDIR`,
`LAUNCHD_DIR`, `LAUNCHD_DOMAIN` — are honored throughout. See the top of
`Makefile` for the full list.

## Configuration

A single TOML file. Pass it via `--config` (or `-c`).

```toml
# Engine telemetry — operator-facing diagnostic stream. The defaults
# emit `info`-level messages to stderr, which a supervisor (systemd,
# launchd, FreeBSD daemon(8)) captures into its log facility.
[log]
level       = "info"                  # trace | debug | info | warn | error
destination = "stderr"                # stderr | file
# path      = "/var/log/specter.log"  # required when destination = "file"

# One [[watch]] block per reaction. Names must be unique. The `actions`
# array names what should run when the watch settles. Each entry runs
# in sequence, stop-on-failure: if step N fails, steps N+1..M don't run
# and the plan reports `Failed`. The engine's outstanding-effect
# accounting is per-plan (one EffectComplete per plan, regardless of
# step count).
[[watch]]
name      = "rebuild"                 # identifies this watch in logs
path      = "/srv/repo/src"           # absolute; pending paths supported
actions   = [{ exec = ["cargo", "build"] }]  # argv-only (no shell expansion)

# Optional knobs (defaults shown).
# enabled    = true                   # false ⇒ inert at runtime; toggle without deleting
settle       = "200ms"                # debounce window after the last event
# max_settle = "1h"                   # forced fire even if events keep arriving (default 1h).
                                      # If `actions` has multiple steps, `max_settle * 4` is
                                      # the recovery budget for the whole plan — tune up for
                                      # long sequences (a 5-step plan that legitimately runs
                                      # 25 min would otherwise trip the hatch).
# scope      = "subtree-root"         # subtree-root | per-stable-file
# events     = ["structure", "content"]  # default mask depends on scope
# pattern    = "**/*.rs"              # glob filter
# exclude    = ["target/**", ".git/**"]
# hidden     = false                  # scan dotfiles
# recursive  = true                   # descend into subdirectories
# max_depth  = 8                      # cap descent depth
# log_output = false                  # forward child stdout/stderr to specter's stdio

[[watch]]
name       = "format-each"
path       = "/srv/repo/docs"
actions    = [{ exec = ["prettier", "--write", "${specter.path}"] }]
scope      = "per-stable-file"        # one Effect per stable file
pattern    = "**/*.md"
log_output = true                     # send formatter output to the journal
```

The `actions` array also accepts the equivalent block-table form, which
reads better for longer watches:

```toml
[[watch]]
name = "format-each"
path = "/srv/repo/docs"

[[watch.actions]]
exec = ["prettier", "--write", "${specter.path}"]
```

### Placeholders

`exec` argv slots reference Specter's substitution catalog through the
`${specter.<name>}` namespace. Anything else `$`-shaped — bare
`$NAME`, `${VAR}`, `$5` — passes through to the spawned process
verbatim, so shell, awk, perl, or make idioms inside an `sh -c` slot
keep working.

Single-value placeholders render one string into the surrounding argv
slot:

| Placeholder           | Meaning                                                                |
|-----------------------|------------------------------------------------------------------------|
| `${specter.path}`     | Absolute path of the target (`per-stable-file`) or anchor (`subtree-root`) |
| `${specter.relative}` | Path relative to the watch anchor (empty for `subtree-root`)           |
| `${specter.anchor}`   | Absolute path of the watch's anchor                                    |
| `${specter.parent}`   | Parent directory of `${specter.path}` (empty only for a subtree-root anchored at `/`) |
| `${specter.watch}`    | Watch name (the `[[watch]] name` field)                                |
| `${specter.time}`     | Wall-clock instant sampled immediately before spawn, RFC 3339 UTC, second-precision (`2026-05-10T12:34:56Z`) |

Multi-value placeholders produce **one argv slot per value**, with any
surrounding literal prefix tiled into each slot; an empty list drops the
entire surrounding slot:

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

A literal **after** a multi-value placeholder (e.g. `"--flag=${specter.excluded}-end"`)
is not appended to each emitted value — when at least one value emits, the trailing
literal becomes its own standalone argv slot. Place per-value suffixes inside the
prefix or use a wrapper script.

Two escape rules round out the grammar:

- `$$` — a literal `$`. Use `$$$$` to pass `$$` through to a shell that
  wants to expand its PID.
- `${specter.<unknown>}` — a typo guard fires (fail-loud) only inside
  the explicit namespace. `${specter.PATH}`, `${specter.}`, and
  unterminated `${specter.path` likewise error during config validation.

Anything that doesn't match `${specter.<name>}` exactly — `$path`,
`${specter}` (no dot), `${SPECTER.path}` (uppercase prefix),
`${HOME}` — is literal pass-through. The shell, if any, sees those
bytes unchanged.

### Environment variables

The spawned child receives a `SPECTER_*` set in addition to the
inherited parent environment:

| Variable                | Value                                                                |
|-------------------------|----------------------------------------------------------------------|
| `SPECTER_PATH`          | mirrors `${specter.path}`                                            |
| `SPECTER_RELATIVE_PATH` | mirrors `${specter.relative}`                                        |
| `SPECTER_ANCHOR`        | mirrors `${specter.anchor}`                                          |
| `SPECTER_PARENT`        | mirrors `${specter.parent}`                                          |
| `SPECTER_WATCH`         | mirrors `${specter.watch}`                                           |
| `SPECTER_TIME`          | mirrors `${specter.time}` (same instant — the resolver samples once per spawn) |
| `SPECTER_EXCLUDED`      | mirrors `${specter.excluded}` (newline-separated, no trailing newline) |
| `SPECTER_CREATED`       | mirrors `${specter.created}` (newline-separated, empty when no diff) |
| `SPECTER_DELETED`       | mirrors `${specter.deleted}` (newline-separated, empty when no diff) |
| `SPECTER_MODIFIED`      | mirrors `${specter.modified}` (newline-separated, empty when no diff) |
| `SPECTER_RENAMED_FROM`  | mirrors `${specter.renamed_from}` (newline-separated, empty when no diff) |
| `SPECTER_RENAMED_TO`    | mirrors `${specter.renamed_to}` (newline-separated, empty when no diff) |
| `SPECTER_EVENT_KIND`    | `dir-subtree` or `file`                                              |
| `SPECTER_FORCED`        | `0` or `1` — `1` when the burst crossed `max_settle` before settling |
| `SPECTER_CORRELATION`   | per-Effect monotonic decimal id                                      |
| `SPECTER_DIFF_PATH`     | absolute path of a tab-separated diff file (set only when the watch's actions reference diff-derived placeholders or `scope = "per-stable-file"`; the same file is shared across every step of a multi-step plan and is removed once the plan exits) |

### CLI flags

```
specter --config <file>          # required
        --log-level <lvl>        # override [log] level
        --log-destination <dst>  # override [log] destination
        --log-path <path>        # override [log] path
        --concurrency <n>        # global Effect spawn cap (default 2 × CPUs)
        --probe-concurrency <n>  # walker pool size (default 4)
```

CLI > config > defaults at every layer. SIGHUP triggers a reload of
the config file; CLI overrides survive.

## Subprocess output

By default, child stdout/stderr go to `/dev/null` — Specter doesn't
parse, format, or annotate user command output. Set `log_output =
true` per watch and the actuator inherits Specter's own stdio fds, so
the supervisor's log facility captures the bytes.

For more elaborate routing (notifications, conditional teeing), use a
shell wrapper:

```toml
actions = [{ exec = ["sh", "-c", "build.sh 2>&1 | tee /tmp/last && curl -d @/tmp/last ntfy.sh/topic"] }]
```

## Layout

```
crates/
  specter-core      # types, snapshot, diff, traits — pure
  specter-engine    # Engine::step — pure, depends only on core
  specter-sensor    # kqueue watcher + worker prober pool
  specter-actuator  # subprocess pool, coalescing, env vars
  specter-config    # TOML + CLI parse / validate / diff
  specter-bin       # wiring, signals, hot reload, drain order
etc/                # systemd / launchd / FreeBSD rc.d templates
Makefile            # build + install conventions
```

## License

Dual-licensed under MIT or Apache-2.0, at your option.
