# Specter install Makefile.
#
# Standard variables:
#   PREFIX      install root (default /usr/local)
#   DESTDIR     staging root for distro packagers (default empty)
#   BINDIR      binary install dir (default $(PREFIX)/bin)
#   SYSCONFDIR  config install dir (default $(PREFIX)/etc)
#   SBINDIR     sbin install dir (default $(PREFIX)/sbin)
#
# Targets:
#   build / install / uninstall / clean        core
#   install-config / uninstall-config          specter.toml example +
#                                              first-install seed
#   install-systemd / install-launchd /
#   install-freebsd                            opt-in service-template
#                                              installs
#   uninstall-systemd / uninstall-launchd /
#   uninstall-freebsd                          service-template removal
#   install-all / uninstall-all                auto-detect host OS

PREFIX     ?= /usr/local
BINDIR     ?= $(PREFIX)/bin
SYSCONFDIR ?= $(PREFIX)/etc
SBINDIR    ?= $(PREFIX)/sbin

INSTALL          ?= install
INSTALL_PROGRAM  ?= $(INSTALL) -m 0755
INSTALL_DATA     ?= $(INSTALL) -m 0644
INSTALL_SCRIPT   ?= $(INSTALL) -m 0755

CARGO ?= cargo
TARGET_RELEASE := target/release/specter

BUILD_OS ?= $(shell uname -s | tr '[:upper:]' '[:lower:]')

SUBST = sed -e 's|@SPECTER_BIN@|$(BINDIR)/specter|g' \
            -e 's|@SPECTER_CONF@|$(SYSCONFDIR)/specter.toml|g'

.PHONY: build install uninstall clean help \
        install-all uninstall-all \
        install-config uninstall-config \
        install-systemd install-launchd install-freebsd \
        uninstall-systemd uninstall-launchd uninstall-freebsd

help:
	@echo "Targets:"
	@echo "  build              cargo build --release"
	@echo "  install            install binary to \$$(BINDIR)"
	@echo "  install-config     install specter.toml.example to \$$(SYSCONFDIR);"
	@echo "                     seeds specter.toml on first install"
	@echo "  install-systemd    install systemd unit to /etc/systemd/system/"
	@echo "  install-launchd    install plist to /Library/LaunchDaemons/"
	@echo "  install-freebsd    install rc.d script to \$$(PREFIX)/etc/rc.d/"
	@echo "  install-all        install binary, config, and host-OS service"
	@echo "                     template (detected: $(BUILD_OS))"
	@echo "  uninstall          remove the installed binary"
	@echo "  uninstall-all      remove binary and host-OS service template"
	@echo "                     (active config files are preserved)"
	@echo
	@echo "Variables: PREFIX=$(PREFIX) DESTDIR=$(DESTDIR) BINDIR=$(BINDIR)"

build:
	$(CARGO) build --release

install: build
	$(INSTALL) -d $(DESTDIR)$(BINDIR)
	$(INSTALL_PROGRAM) $(TARGET_RELEASE) $(DESTDIR)$(BINDIR)/specter

uninstall:
	rm -f $(DESTDIR)$(BINDIR)/specter

clean:
	$(CARGO) clean

install-config:
	$(INSTALL) -d $(DESTDIR)$(SYSCONFDIR)
	$(INSTALL_DATA) etc/specter.toml.example \
	    $(DESTDIR)$(SYSCONFDIR)/specter.toml.example
	@if [ ! -e $(DESTDIR)$(SYSCONFDIR)/specter.toml ]; then \
	    $(INSTALL_DATA) etc/specter.toml.example \
	        $(DESTDIR)$(SYSCONFDIR)/specter.toml; \
	    echo "Seeded $(SYSCONFDIR)/specter.toml from example."; \
	else \
	    echo "Preserved existing $(SYSCONFDIR)/specter.toml."; \
	fi

uninstall-config:
	rm -f $(DESTDIR)$(SYSCONFDIR)/specter.toml.example
	@echo "Left $(SYSCONFDIR)/specter.toml in place; remove manually if desired."

install-systemd:
	$(INSTALL) -d $(DESTDIR)/etc/systemd/system
	$(SUBST) etc/systemd/specter.service \
	    > $(DESTDIR)/etc/systemd/system/specter.service
	chmod 0644 $(DESTDIR)/etc/systemd/system/specter.service
	@echo "Next: systemctl daemon-reload && systemctl enable --now specter"

uninstall-systemd:
	rm -f $(DESTDIR)/etc/systemd/system/specter.service

install-launchd:
	$(INSTALL) -d $(DESTDIR)/Library/LaunchDaemons
	$(SUBST) etc/launchd/io.specter.plist \
	    > $(DESTDIR)/Library/LaunchDaemons/io.specter.plist
	chmod 0644 $(DESTDIR)/Library/LaunchDaemons/io.specter.plist
	@echo "Next: sudo launchctl load -w /Library/LaunchDaemons/io.specter.plist"

uninstall-launchd:
	rm -f $(DESTDIR)/Library/LaunchDaemons/io.specter.plist

install-freebsd:
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/etc/rc.d
	$(SUBST) etc/freebsd/specter \
	    > $(DESTDIR)$(PREFIX)/etc/rc.d/specter
	chmod 0755 $(DESTDIR)$(PREFIX)/etc/rc.d/specter
	@echo "Next: add 'specter_enable=\"YES\"' to /etc/rc.conf, then service specter start"

uninstall-freebsd:
	rm -f $(DESTDIR)$(PREFIX)/etc/rc.d/specter

install-all: install install-config
	@case "$(BUILD_OS)" in \
	  linux)   $(MAKE) install-systemd ;; \
	  darwin)  $(MAKE) install-launchd ;; \
	  freebsd) $(MAKE) install-freebsd ;; \
	  *) echo "specter: unknown OS '$(BUILD_OS)'; binary + config installed, service template skipped" ; \
	     echo "         run one of: make install-systemd | install-launchd | install-freebsd" ;; \
	esac

uninstall-all: uninstall
	@case "$(BUILD_OS)" in \
	  linux)   $(MAKE) uninstall-systemd ;; \
	  darwin)  $(MAKE) uninstall-launchd ;; \
	  freebsd) $(MAKE) uninstall-freebsd ;; \
	  *) echo "specter: unknown OS '$(BUILD_OS)'; binary removed, service template skipped" ;; \
	esac
	@echo "Left $(SYSCONFDIR)/specter.toml{,.example} in place; run uninstall-config to remove the .example."
