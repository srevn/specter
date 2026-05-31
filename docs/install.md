# Install

```sh
make build               # cargo build --release
make install             # → $(BINDIR)/specter
make install-config      # seeds $(SYSCONFDIR)/specter.toml from etc/specter.toml.example
make install-all         # binary + config + host-OS service template (auto-detect)
```

> FreeBSD: use `gmake` (base `make` is BSD make). macOS and Linux: `make` is GNU make.

Individual service templates:

```sh
make install-systemd     # $(SYSTEMD_DIR)/specter.service
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
| `SYSTEMD_DIR`     | `$(PREFIX)/lib/systemd/system` (Linux) · set `/etc/systemd/system` for admin-owned |
| `LAUNCHD_DIR`     | `~/Library/LaunchAgents` (macOS user) · `/Library/LaunchDaemons` (macOS system) |
| `LAUNCHD_DOMAIN`  | `gui/$(id -u)` (macOS user) · `system` (macOS system) |
| `DESTDIR`         | empty (staging-prefix for packaging)                 |
| `SPECTER_USER`    | invoking user (`$SUDO_USER` under sudo)              |
| `SPECTER_GROUP`   | invoking user's primary group                        |
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
- **IPC socket** — the daemon binds `/run/specter/specter.sock`
  (systemd provisions the dir via `RuntimeDirectory=specter`),
  `/var/run/specter/specter.sock` (the FreeBSD rc script provisions it
  in `start_precmd` and passes `--socket`), or the fixed
  `/tmp/specter.sock` (launchd, macOS). Clients resolve the same path
  and need no flag; override with `--socket <path>` or `$SPECTER_SOCK`.

## Uninstall

```sh
make uninstall           # binary only
make uninstall-all       # binary + host-OS service template (config preserved)
make uninstall-config    # removes $(SYSCONFDIR)/specter.toml
```

`uninstall-all` deliberately leaves `$(SYSCONFDIR)/specter.toml` in place —
operator config is preserved across reinstalls.
