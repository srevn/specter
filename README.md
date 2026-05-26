# Specter

> Prove the absence of change.

Specter watches paths and fires commands when the tree settles — not
when "something happened," but when **nothing has happened for long
enough** that the tree is observably stable. Commands run against a
snapshot that includes every change up to the quiescent point.

Conventional file-watch tools fire on every kernel event. The result
is a flurry of redundant runs against partially-written files. Specter
inverts that contract:

- **Coarse file-tree settling** — no double-fires on `git checkout`,
  multi-file editor saves, or build outputs writing dozens of artifacts.
- **Hierarchical content hashing** — re-running the same edit (saving
  with no changes, touching mtime, idempotent reformatters) does not
  re-fire the command.
- **Self-event absorption** — the reaction itself usually writes inside
  the watched tree; Specter folds those events into the post-fire rebase
  rather than treating them as a fresh burst.
- **Hot config reload** — SIGHUP, an operator IPC verb, or
  edit-and-save against a watched config path.
- **Operator control surface** — eight client verbs over a UNIX socket
  (`status`, `list`, `show`, `disable`, `enable`, `reload`, `tail`,
  `wait`) for live inspection and runtime overrides.

Under the hood, a pure engine drives a kqueue/inotify sensor (BSD,
macOS, Linux) and a subprocess actuator over bounded channels.

## Quick start

```sh
make install-all                 # binary + config + host-OS service template
$EDITOR /usr/local/etc/specter.toml
# launchd / systemd / FreeBSD rc.d will start the daemon
specter status                   # confirm it's up
```

See [docs/install.md](docs/install.md) for variables, scopes, and
service-template details.

## A watch

```toml
[[watch]]
name    = "rebuild"
path    = "/srv/repo/src"
actions = [{ exec = ["cargo", "build"] }]
```

Each `[[watch]]` block declares one reaction: a name, an absolute path,
and an `actions` array. Optional knobs cover settle window, scope
(subtree vs per-file), glob filters, event mask, recursion, and child
stdio routing. The actions array supports sequences, pipes (`a | b | c`
with pipefail-on), and conditionals (`when` / `then` / `else`). Argv
slots accept `${specter.*}` and `${env.*}` placeholders; the child
process receives a matching `SPECTER_*` environment set.

See:

- [docs/configuration.md](docs/configuration.md) — TOML reference,
  actions, pipes, conditionals, scope, events.
- [docs/placeholders.md](docs/placeholders.md) — placeholder catalog,
  environment substitution, child-process env vars.

## Commands

```
specter run --config <file>         # the daemon (typically run by the supervisor)
specter status                      # daemon snapshot
specter list                        # every watch + state
specter show <name>                 # one watch in detail
specter disable <name>              # runtime override (survives reload)
specter enable <name>               # clear the runtime override
specter reload                      # equivalent to SIGHUP
specter tail [--filter <tag> …]     # stream diagnostics
specter wait <name> [--timeout …]   # block until the watch fires (or detaches)
```

`specter --help` prints the full surface; `--socket <path>` overrides
the default UNIX socket on every client verb.

See [docs/control.md](docs/control.md) for the IPC reference — socket
path defaults, wire format, error codes, subscribe semantics, exit
codes.

## Subprocess output

Child stdout/stderr go to `/dev/null` by default — Specter doesn't
parse, format, or annotate user command output. Set `log_output = true`
per watch and the actuator inherits Specter's own stdio fds, so the
supervisor's log facility captures the bytes. For richer routing
(notifications, conditional teeing), wrap the action with `sh -c`.

## Layout

```
crates/
  specter-core      # types, snapshot, diff, traits — pure
  specter-engine    # Engine::step — pure, depends only on core
  specter-sensor    # kqueue/inotify watcher + worker prober pool
  specter-actuator  # subprocess pool, coalescing, env vars
  specter-config    # TOML + CLI parse / validate / diff
  specter-bin       # wiring, signals, hot reload, IPC surface
etc/                # systemd / launchd / FreeBSD rc.d templates
docs/               # operator-facing reference
Makefile            # build + install conventions
```

## License

Dual-licensed under MIT or Apache-2.0, at your option.
