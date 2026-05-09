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
inverts the contract: events restart a settle timer; commands fire
**only after the burst has decayed**, against a snapshot of the tree
that includes every change up to that quiescent point.

Concretely:

- Coarse file-tree settling — no double-fires on `git checkout`,
  multi-file editor saves, or build outputs writing dozens of
  artifacts.
- Hierarchical content hashing — re-running the same edit (saving
  with no changes, touching mtime, idempotent reformatters) does not
  re-fire the command.
- Self-event absorption — the command itself usually writes inside
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
make install-launchd     # → /Library/LaunchDaemons/io.specter.plist
make install-freebsd     # → $(PREFIX)/etc/rc.d/specter

# All-in-one — auto-detect host OS and pick the right service template:
make install-all
```

Standard variables — `PREFIX`, `DESTDIR`, `BINDIR`, `SYSCONFDIR` — are
honored throughout. See the top of `Makefile` for the full list.

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

# One [[watch]] block per command. Names must be unique.
[[watch]]
name      = "rebuild"                 # identifies this watch in logs
path      = "/srv/repo/src"           # absolute; pending paths supported
command   = ["cargo", "build"]        # argv-only (no shell expansion)

# Optional knobs (defaults shown).
# enabled    = true                   # false ⇒ inert at runtime; toggle without deleting
settle       = "200ms"                # debounce window after the last event
# max_settle = "1h"                   # forced fire even if events keep arriving (default 1h)
# scope      = "subtree-root"         # subtree-root | per-stable-file
# events        = ["structure", "content"]  # default mask depends on scope
# pattern       = "**/*.rs"           # glob filter
# exclude       = ["target/**", ".git/**"]
# hidden        = false               # scan dotfiles
# recursive     = true                # descend into subdirectories
# max_depth     = 8                   # cap descent depth
# log_output    = false               # forward child stdout/stderr to specter's stdio

[[watch]]
name       = "format-each"
path       = "/srv/repo/docs"
command    = ["prettier", "--write", "$path"]
scope      = "per-stable-file"        # one Effect per stable file
pattern    = "**/*.md"
log_output = true                     # send formatter output to the journal
```

### Placeholders

`command` slots may reference Specter's resolved-substitution catalog:

| Placeholder | Meaning                                                 |
|-------------|---------------------------------------------------------|
| `$path`     | Absolute path of the target (`per-stable-file`) or anchor (`subtree-root`) |
| `$rel`      | Path relative to the watch anchor                       |
| `$anchor`   | Absolute path of the watch's anchor                     |
| `$created`  | Newline-separated list of created paths                 |
| `$deleted`  | Newline-separated list of deleted paths                 |
| `$modified` | Newline-separated list of modified paths                |
| `$renamedfrom`, `$renamedto` | Rename pairs                           |

Plus environment variables — `$HOME`, `$USER`, `$SPECTER_*` — pass
through unchanged for the spawned shell to expand.

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
command = ["sh", "-c", "build.sh 2>&1 | tee /tmp/last && curl -d @/tmp/last ntfy.sh/topic"]
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
