# Control surface

A running daemon exposes a UNIX-socket control plane for operators. Nine
client verbs ship inside the same `specter` binary: read-only inspection
(`status`, `list`, `show`), runtime mutation (`disable`, `enable`,
`absorb`, `reload`), and streaming (`tail`, `wait`).

```sh
specter status              # daemon snapshot
specter list                # every watch, and each match of a dynamic one
specter show <name>         # one watch in detail
specter disable <name>      # runtime override; persists across reload
specter enable <name>       # clear the runtime override
specter absorb <name>       # fold the next change into the baseline, don't fire
specter reload              # equivalent to SIGHUP
specter tail                # stream every diagnostic
specter wait <name>         # block until the named watch fires (or detaches)
```

A dynamic watch — a `path` holding a glob, see
[configuration.md](configuration.md#dynamic-watches) — holds one watch per
match, each listed as `<name>@<path>`. The verbs take the dynamic watch's own
name and reach every match it holds when they are asked. A match's own name
is refused by `disable` and `absorb`, unknown to `show`, and taken by `wait`.
On the wire the dynamic watch's Sub is its *template*, and each match's Sub
is *minted* by it: lines about a match carry `minted_by=<template>`.

## The socket

Daemon and client resolve the socket path from one policy, so a client
needs no flag to reach a daemon started with none. Precedence:
`--socket <path>` > `$SPECTER_SOCK` > the per-platform convention:

| Platform | Convention                                                                          |
|----------|-------------------------------------------------------------------------------------|
| Linux    | `$XDG_RUNTIME_DIR/specter.sock` (session) if set, else `/run/specter/specter.sock` (system) |
| macOS    | `/tmp/specter.sock`                                                                  |
| BSD      | `/var/run/specter/specter.sock`                                                     |

The daemon binds one path; the client probes the convention in order —
on Linux the session socket first, then the system socket — and connects
to the first that answers. An explicit `--socket` / `$SPECTER_SOCK` is
pinned with no fall-through: a stale override is reported, never silently
retargeted to a different daemon.

Permissions are `0600` — owner-only by construction via an atomic-rename
bind (`bind(temp)` → `chmod 0600` → `rename(temp, path)`), so the
operator-facing path never appears at a more permissive mode. The
daemon detects and recovers from stale socket files left behind by a
crashed predecessor.

## Wire format

JSONL over the UNIX socket: one request per line, one response per line,
newline-delimited. Every parse is strict — unknown fields and unknown
verb tags are rejected (`malformed` error code) rather than silently
ignored. The protocol surface lives entirely inside `specter-bin`; the
in-tree client is the reference consumer.

## Verbs

### `status`

Snapshot of daemon state:

```sh
$ specter status
specter status
─────────────────────────────────────────────────────────────
uptime              4:17:02
started             2026-05-26T08:12:33Z
reloads             3 (last 2026-05-26T11:42:08Z via ipc)
subs                12 attached · 2 disabled (toml) · 1 disabled (runtime)
profiles            3 active
discovery           1 template
config              /etc/specter.toml
socket              /run/user/1000/specter.sock
```

`subs … attached` counts every Sub the daemon holds: each static watch,
each dynamic watch, and each of its matches. `profiles … active` counts the
Profiles at work — a burst in flight, or a pending path awaited.
`discovery` counts the dynamic watches attached.

`-o json` emits the lossless wire shape. `--wide` is accepted and changes
nothing: `status` has no wider view.

### `list`

Every attached watch, each match of a dynamic watch as its own
`<name>@<path>` row, and every runtime-disabled and TOML-disabled watch —
one row per name, alphabetically. `NAME`, `STATE`, `ANCHOR`, `LAST_FIRED`,
`FIRES`, `REACTION`, `DISABLED` columns; `--wide` adds `PROFILE_ID`,
`SUB_ID`, `DEDUP_SUPPRESSED`, `SETTLE`. `-o json` carries the lossless
`ListRow` shape, where a match's row names its template in `minted_by`.

`REACTION` is `spawn` for a watch or a match that runs actions, and `mint`
for a dynamic watch's own row: it runs nothing itself, so its fire columns
stay `-`.

A row's `DISABLED` column distinguishes the two disable sources:

- `runtime` — operator ran `specter disable`; cleared by `enable` or by
  removing the entry from TOML entirely.
- `toml` — `[[watch]]` carries `enabled = false`; cleared by editing
  TOML and reloading.

### `show <name>`

One watch in detail. Three outcomes:

- `active` — the full `SubDetails` block: state, anchor, last-fired,
  fire / suppress / absorb counters, any live `absorb` window, scope,
  action program rendering. For a dynamic watch the block says what each
  match runs (`minted settle`, `minted scope`, `minted program`), how many
  matches it holds (`minted live`), and while any of them is absorbing, how
  many and until when (`absorbing  2 of 3 matches, until <T> (persist)`).
- `disabled` — the watch is operator-declared but not attached; the
  response carries the `source` (`runtime` / `toml`) so the operator
  knows what to do next.
- `unknown` — no `[[watch]]` block and no runtime disable record. A match's
  own name answers `unknown` too: `list` shows the matches.

### `disable <name>`

Detach a running watch and add `<name>` to the daemon's runtime-disable
set. Survives reload as long as the `[[watch]]` entry stays in TOML; if
the operator removes the entry, the runtime override is implicitly
cleared on the next reload.

On a dynamic watch it detaches the watch and every match with it; each
match's `sub_detached` carries `reason=template_detached`. A match's own
name is refused with `minted_sub_no_op`: the dynamic watch would find the
match again at its next walk, so address the watch by its own name. A
watch already disabled at runtime answers `not_disabled`; a name no attached
watch holds, `unknown_sub`.

### `enable <name>`

Clear the runtime override and re-attach the watch. Two failure modes:

- `not_disabled` — `<name>` is not in the runtime-disable set (a match's
  name never is).
- `toml_disabled` — the override clears, but the watch is also
  TOML-disabled (or missing from the file). Edit the config and reload
  to attach.

A dynamic watch enabled again finds its matches afresh; being present when
it attaches, they do not run its actions.

### `absorb <name>`

Arm a **fold-without-fire window**: the next change Specter would fire on
is instead folded silently into the baseline, advancing the settled
reference without running the watch's actions. The signal for an
*expected* change — most often a replication echo.

The motivating case is two daemons watching the same tree on two hosts.
A's fire `rsync`s the result to B; with no hint, B observes the rsync and
fires its own redundant reaction. Sent over the same channel as the
rsync, just ahead of it, `specter absorb <name>` tells B that the next
change is that expected replication: absorb it, don't echo.

A fold **advances** the baseline — it does not merely suppress one fire —
so detection stays correct afterwards: a later, genuine change still
fires.

```sh
specter absorb my-watch              # fold the next single change
specter absorb my-watch --for 30s    # fold every change for the next 30s
```

- **Bare `absorb`** arms a one-shot window: it folds the first change
  that would have fired, then retires. The window is one settle interval
  wide, so the expected change must begin within roughly one settle of
  the signal; if a fold-worthy change never arrives, the window lapses on
  its own and nothing is folded.
- **`absorb --for <dur>`** holds the window open for `dur` (units up to
  hours — `500ms`, `30s`, `1m30s`, `2h`), folding every change that would
  have fired in that span. Reach for it when the replication starts after
  a setup delay longer than the settle window, or spans a run of separate
  bursts.

A window is **per-watch-Profile**: watches that share an anchor path and
scan config fold together (the same grouping that shares a settle
lifecycle), and the fold counter is likewise per-Profile. Re-arming
replaces the current window outright.

While a window is live, `show <name>` adds an `absorbing until <T>
(<mode>)` line and the `fires` line carries a running `absorbed: <n>`
counter. On the `tail` stream an arm emits `absorb_armed` and each fold
emits `quiescence_absorbed`.

On a dynamic watch, `absorb` arms a window on each match it holds when it
is asked — one `absorb_armed` each — and `show` reports how many are
absorbing and until when. A match that appears afterwards is not covered,
and a dynamic watch that holds no match arms nothing and still answers ok.

A match's own name is refused with `minted_sub_no_op`, as by `disable`; a
name no attached watch holds returns `unknown_sub`. Like the other mutating
verbs it is refused with `shutting_down` once the daemon has begun a
graceful exit.

### `reload`

Re-read the config file and apply the diff. Equivalent to `SIGHUP`,
attributed differently in `status`'s `last_reload_via`. The reply
blocks until apply completes — operators reading `status` immediately
after `reload` see the new state.

### `tail`

Stream every `WireDiagnostic` from the moment the subscription is
acknowledged. Each line is a JSON object keyed by `diag` (the variant
tag). Default render is `-o human` (one compact line per event);
`-o json` emits the raw wire shape.

```sh
specter tail
specter tail --filter sub_fired
specter tail --filter sub_fired --filter sub_detached
specter tail -o json | jq 'select(.diag == "sub_fired")'
```

`--filter` validates against the closed wire vocabulary at handler
entry — an unknown tag exits `2` with the full list printed to stderr.
Filtering is client-side (the daemon streams the full set; the client
discards non-matches). `tail` takes no watch name: a stream scoped to one
watch is `wait`'s, and the wire's (see *Subscribe semantics*).

Back-pressure: when a slow reader can't drain fast enough, the daemon
drops events into a per-conn counter and emits a `_missed` marker on
the next successful write (`{"diag":"_missed","count":N,"at":"..."}`).
The marker appears in-order so operators see exactly where the gap is.

Exit codes:
- `0` — graceful EOF (daemon shutdown or downstream pipe closed)
- `1` — connect / subscribe / read failure
- `2` — unknown `--filter` tag

### `wait <name>`

Block until `<name>` fires (default) or detaches (`--kind detach`).
One round-trip: the daemon resolves `name → SubId` atomically with
the subscribe, closing the race between "show says it exists" and
"subscribe attaches".

```sh
specter wait my-watch                          # block until next fire
specter wait my-watch --timeout 30s            # bounded wait
specter wait my-watch --kind detach            # block until detach
```

On a dynamic watch, `wait` returns at the first fire of any of its matches.
Only the watch's own detach decides: `--kind fire` exits `2` when the watch
itself leaves, not when a match is reaped, and `--kind detach` waits for
the watch. A match's own name waits on that match alone.

Exit codes:
- `0` — matched the requested kind
- `1` — connect / subscribe failure (including `unknown_sub`), or the
  daemon closed the stream before a match
- `2` — `--kind fire` but the watch detached before firing (no fire is
  coming)
- `124` — timeout elapsed (POSIX `timeout(1)` convention)

## Output formats

`status`, `list`, and `show` accept `-o human` (default, table-style)
and `-o json` (lossless wire shape, one object). `tail` accepts the
same flags but for the per-event line shape; `wait` always renders the
matching event with the human renderer.

`tail -o json` and the daemon's emission share one source — re-parsing
the human client's JSON output yields the same in-memory structure the
daemon serialized.

## Error codes

A failed verb returns a structured error:

```json
{"kind":"err","code":"toml_disabled","error":"runtime override cleared, but the watch is not active in the current TOML (enabled = false or removed); edit config and reload to re-attach"}
```

| Code                 | Meaning                                                          |
|----------------------|------------------------------------------------------------------|
| `unknown_sub`        | The verb named no attached watch: a typo, a TOML-disabled watch, or — for `absorb` and `wait` — one disabled at runtime. |
| `minted_sub_no_op`   | `disable` or `absorb` named a match of a dynamic watch; address the watch by its own name. |
| `not_disabled`       | `enable` against a watch not in the runtime-disable set, or `disable` against one already in it. |
| `toml_disabled`      | `enable` cleared the override but the watch is TOML-disabled.    |
| `busy`               | Connection cap reached.                                          |
| `response_too_big`   | Serialized response exceeds the per-conn write cap.              |
| `malformed`          | Request line failed JSON parse or carried an unknown `op`.       |
| `already_subscribed` | `Subscribe` invoked on a conn already in subscriber mode.        |
| `shutting_down`      | Daemon is winding down; mutating verbs refused, reads still work.|

Read-only verbs (`status`, `list`, `show`, `tail`, `wait`) keep working
during graceful shutdown so operators can observe the daemon's exit.

## Subscribe semantics

`tail` and `wait` both flip the connection into **subscriber mode** —
one-shot per conn (a second `Subscribe` returns `already_subscribed`).
`wait` carries a `name` field; the daemon resolves it server-side
before acknowledging, and the ack names the Sub it resolved
(`{"kind":"subscribe_ack","sub":<id>}`). `tail` carries no name and
receives every event.

A subscription by name receives the lines about that watch's Sub and, for
a dynamic watch, the lines about each of its matches. A line that names
only a Profile (`absorb_armed`, `quiescence_absorbed`, `probe_failed`, …)
reaches an unnamed subscription alone. The subscription keeps the Sub it
resolved: a watch replaced by a config edit goes quiet on it.

Subscriber registration happens **before** the ack reply is written —
no diagnostic emitted between request arrival and ack delivery is
lost. The pattern is "ack-before-stream": the operator sees the
subscribe complete only after the daemon has wired them into the
broker's fanout list.
