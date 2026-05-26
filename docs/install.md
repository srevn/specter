# Install

```sh
make build               # cargo build --release
make install             # → $(BINDIR)/specter
make install-config      # seeds $(SYSCONFDIR)/specter.toml from etc/specter.toml.example
make install-all         # binary + config + host-OS service template (auto-detect)
```

Individual service templates:

```sh
make install-systemd     # /etc/systemd/system/specter.service
make install-launchd     # $(LAUNCHD_DIR)/io.specter.plist
make install-freebsd     # $(PREFIX)/etc/rc.d/specter
```

Each command prints the platform-specific follow-up (`systemctl daemon-reload`,
`launchctl bootstrap`, `service specter start`).

## Variables

Standard GNU-style overrides honored throughout:

| Variable          | Default                                              |
|-------------------|------------------------------------------------------|
| `PREFIX`          | `/usr/local`                                         |
| `BINDIR`          | `$(PREFIX)/bin`                                      |
| `SYSCONFDIR`      | `$(PREFIX)/etc` (Linux/BSD) · `~/.config/specter` (macOS) |
| `LAUNCHD_DIR`     | `~/Library/LaunchAgents` (macOS user) · `/Library/LaunchDaemons` (macOS system) |
| `LAUNCHD_DOMAIN`  | `gui/$(id -u)` (macOS user) · `system` (macOS system) |
| `DESTDIR`         | empty (staging-prefix for packaging)                 |
| `SPECTER_USER`    | invoking user                                        |
| `SPECTER_GROUP`   | invoking group                                       |
| `SPECTER_LOG_DIR` | `~/Library/Logs` (macOS) · `/var/log` (others)       |

`make help` prints the resolved values for the current invocation.

## macOS — user vs system scope

The defaults install at **user scope** — no `sudo` needed, plist lands in
`~/Library/LaunchAgents`, config in `~/.config/specter`, and
`launchctl bootstrap`/`bootout` use `gui/<uid>`. For a system install,
override explicitly with sudo:

```sh
sudo make install-launchd \
    LAUNCHD_DIR=/Library/LaunchDaemons \
    LAUNCHD_DOMAIN=system \
    SYSCONFDIR=/usr/local/etc
```

Linux and FreeBSD service installs are system-scope only.

## Service template behavior

All three templates run `specter run --config <path>` under the supervisor;
they share a common reload contract:

- **Reload** — `SIGHUP` re-reads the config file in-place. systemd's
  `ExecReload=` and FreeBSD's `service specter reload` both send SIGHUP.
  On launchd, `launchctl kill -s HUP <Label>` does the same. The IPC
  `specter reload` verb is equivalent.
- **Restart on failure** — systemd: `Restart=on-failure` with 5s backoff.
  launchd: `KeepAlive=true`. FreeBSD: managed by `daemon(8)`.
- **Logs** — go to the supervisor's journal (systemd), the configured
  `StandardOutPath` (launchd), or the file in `command_args -o ...`
  (FreeBSD). Specter's own log destination is set in `[log]`.

## Uninstall

```sh
make uninstall           # binary only
make uninstall-all       # binary + host-OS service template (config preserved)
make uninstall-config    # removes $(SYSCONFDIR)/specter.toml
```

`uninstall-all` deliberately leaves `$(SYSCONFDIR)/specter.toml` in place —
operator config is preserved across reinstalls.
