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
#   build / install / uninstall / clean       core
#   install-systemd / install-launchd /
#   install-freebsd                           opt-in service-template installs
#   install-all                               auto-detect host OS + install
#                                             the matching service template

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

.PHONY: build install uninstall clean help install-all \
        install-systemd install-launchd install-freebsd

help:
	@echo "Targets:"
	@echo "  build              cargo build --release"
	@echo "  install            install binary to \$$(BINDIR)"
	@echo "  install-systemd    install systemd unit to /etc/systemd/system/"
	@echo "  install-launchd    install plist to /Library/LaunchDaemons/"
	@echo "  install-freebsd    install rc.d script to \$$(PREFIX)/etc/rc.d/"
	@echo "  install-all        install binary + service template for host OS"
	@echo "                     (detected: $(BUILD_OS))"
	@echo "  uninstall          remove the installed binary"
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

install-systemd:
	$(INSTALL) -d $(DESTDIR)/etc/systemd/system
	$(INSTALL_DATA) etc/systemd/specter.service \
	    $(DESTDIR)/etc/systemd/system/specter.service
	@echo "Next: systemctl daemon-reload && systemctl enable --now specter"

install-launchd:
	$(INSTALL) -d $(DESTDIR)/Library/LaunchDaemons
	$(INSTALL_DATA) etc/launchd/io.specter.plist \
	    $(DESTDIR)/Library/LaunchDaemons/io.specter.plist
	@echo "Next: sudo launchctl load -w /Library/LaunchDaemons/io.specter.plist"

install-freebsd:
	$(INSTALL) -d $(DESTDIR)$(PREFIX)/etc/rc.d
	$(INSTALL_SCRIPT) etc/freebsd/specter \
	    $(DESTDIR)$(PREFIX)/etc/rc.d/specter
	@echo "Next: add 'specter_enable=\"YES\"' to /etc/rc.conf, then service specter start"

install-all: install
	@case "$(BUILD_OS)" in \
	  linux)   $(MAKE) install-systemd ;; \
	  darwin)  $(MAKE) install-launchd ;; \
	  freebsd) $(MAKE) install-freebsd ;; \
	  *) echo "specter: unknown OS '$(BUILD_OS)'; binary installed, service template skipped" ; \
	     echo "         run one of: make install-systemd | install-launchd | install-freebsd" ;; \
	esac
