# Specter install Makefile
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

MAKEFLAGS += --no-print-directory

PREFIX  ?= /usr/local
BINDIR  ?= $(PREFIX)/bin
SBINDIR ?= $(PREFIX)/sbin

BUILD_OS ?= $(shell uname -s | tr '[:upper:]' '[:lower:]')

ifeq ($(BUILD_OS),darwin)
    SYSCONFDIR      ?= $(HOME)/.config/specter
    LAUNCHD_DIR     ?= $(HOME)/Library/LaunchAgents
    LAUNCHD_DOMAIN  ?= gui/$(shell id -u)
    SPECTER_LOG_DIR ?= $(HOME)/Library/Logs
else
    LAUNCHD_DIR     ?= /Library/LaunchDaemons
    LAUNCHD_DOMAIN  ?= system
    SPECTER_LOG_DIR ?= /var/log
endif

SYSCONFDIR    ?= $(PREFIX)/etc
SPECTER_LOG   ?= $(SPECTER_LOG_DIR)/specter.log
SPECTER_USER  ?= $(shell id -un)
SPECTER_GROUP ?= $(shell id -gn)

INSTALL          ?= install
INSTALL_PROGRAM  ?= $(INSTALL) -m 0755
INSTALL_DATA     ?= $(INSTALL) -m 0644
INSTALL_SCRIPT   ?= $(INSTALL) -m 0755

CARGO ?= cargo
TARGET_RELEASE := target/release/specter

SUBST = sed -e 's|@SPECTER_BIN@|$(BINDIR)/specter|g' \
            -e 's|@SPECTER_CONF@|$(SYSCONFDIR)/specter.toml|g' \
            -e 's|@SPECTER_LOG@|$(SPECTER_LOG)|g' \
            -e 's|@SPECTER_USER@|$(SPECTER_USER)|g' \
            -e 's|@SPECTER_GROUP@|$(SPECTER_GROUP)|g'

.PHONY: build install uninstall clean help \
        install-all uninstall-all \
        install-config uninstall-config \
        install-systemd install-launchd install-freebsd \
        uninstall-systemd uninstall-launchd uninstall-freebsd

help:
	@echo "Targets:"
	@echo "  build              cargo build --release"
	@echo "  install            install binary to \$$(BINDIR)"
	@echo "  install-config     install specter.toml to \$$(SYSCONFDIR)"
	@echo "                     (preserves any existing file)"
	@echo "  install-systemd    install systemd unit to /etc/systemd/system/"
	@echo "  install-launchd    install plist to \$$(LAUNCHD_DIR)"
	@echo "  install-freebsd    install rc.d script to \$$(PREFIX)/etc/rc.d/"
	@echo "  install-all        install binary, config, and host-OS service"
	@echo "                     template (detected: $(BUILD_OS))"
	@echo "  uninstall          remove the installed binary"
	@echo "  uninstall-all      remove binary and host-OS service template"
	@echo "                     (active config files are preserved)"
	@echo
	@echo "Variables: PREFIX=$(PREFIX) DESTDIR=$(DESTDIR) BINDIR=$(BINDIR)"
	@echo "           SYSCONFDIR=$(SYSCONFDIR) LAUNCHD_DIR=$(LAUNCHD_DIR)"

build:
	@$(CARGO) build --release

install: build
	@echo "Installing specter to $(DESTDIR)$(BINDIR)/"
	@$(INSTALL) -d $(DESTDIR)$(BINDIR)
	@$(INSTALL_PROGRAM) $(TARGET_RELEASE) $(DESTDIR)$(BINDIR)/specter

uninstall:
	@echo "Removing $(DESTDIR)$(BINDIR)/specter"
	@rm -f $(DESTDIR)$(BINDIR)/specter

clean:
	@$(CARGO) clean

install-config:
	@$(INSTALL) -d $(DESTDIR)$(SYSCONFDIR)
	@if [ ! -e $(DESTDIR)$(SYSCONFDIR)/specter.toml ]; then \
	    $(INSTALL_DATA) etc/specter.toml.example \
	        $(DESTDIR)$(SYSCONFDIR)/specter.toml; \
	    echo "Installed $(SYSCONFDIR)/specter.toml"; \
	else \
	    echo "Preserved existing $(SYSCONFDIR)/specter.toml"; \
	fi

uninstall-config:
	@echo "Removing $(DESTDIR)$(SYSCONFDIR)/specter.toml"
	@rm -f $(DESTDIR)$(SYSCONFDIR)/specter.toml

install-systemd:
	@echo "Installing systemd unit to $(DESTDIR)/etc/systemd/system/specter.service"
	@$(INSTALL) -d $(DESTDIR)/etc/systemd/system
	@$(SUBST) etc/systemd/specter.service \
	    > $(DESTDIR)/etc/systemd/system/specter.service
	@chmod 0644 $(DESTDIR)/etc/systemd/system/specter.service
	@echo "Next: systemctl daemon-reload && systemctl enable --now specter"

uninstall-systemd:
	@if [ -z "$(DESTDIR)" ]; then \
	    echo "Stopping and disabling specter"; \
	    systemctl disable --now specter 2>/dev/null || true; \
	fi
	@echo "Removing $(DESTDIR)/etc/systemd/system/specter.service"
	@rm -f $(DESTDIR)/etc/systemd/system/specter.service
	@if [ -z "$(DESTDIR)" ]; then \
	    systemctl daemon-reload 2>/dev/null || true; \
	fi

install-launchd:
	@echo "Installing plist to $(DESTDIR)$(LAUNCHD_DIR)/io.specter.plist"
	@$(INSTALL) -d $(DESTDIR)$(LAUNCHD_DIR) $(DESTDIR)$(SPECTER_LOG_DIR)
	@$(SUBST) etc/launchd/io.specter.plist \
	    > $(DESTDIR)$(LAUNCHD_DIR)/io.specter.plist
	@chmod 0644 $(DESTDIR)$(LAUNCHD_DIR)/io.specter.plist
	@echo "Next: launchctl bootstrap $(LAUNCHD_DOMAIN) $(LAUNCHD_DIR)/io.specter.plist"

uninstall-launchd:
	@if [ -z "$(DESTDIR)" ]; then \
	    echo "Unloading specter from $(LAUNCHD_DOMAIN)"; \
	    launchctl bootout $(LAUNCHD_DOMAIN) $(LAUNCHD_DIR)/io.specter.plist 2>/dev/null || true; \
	fi
	@echo "Removing $(DESTDIR)$(LAUNCHD_DIR)/io.specter.plist"
	@rm -f $(DESTDIR)$(LAUNCHD_DIR)/io.specter.plist

install-freebsd:
	@echo "Installing rc.d script to $(DESTDIR)$(PREFIX)/etc/rc.d/specter"
	@$(INSTALL) -d $(DESTDIR)$(PREFIX)/etc/rc.d
	@$(SUBST) etc/freebsd/specter \
	    > $(DESTDIR)$(PREFIX)/etc/rc.d/specter
	@chmod 0755 $(DESTDIR)$(PREFIX)/etc/rc.d/specter
	@echo "Next: add 'specter_enable=\"YES\"' to /etc/rc.conf, then service specter start"

uninstall-freebsd:
	@if [ -z "$(DESTDIR)" ]; then \
	    echo "Stopping specter service"; \
	    service specter onestop 2>/dev/null || true; \
	fi
	@echo "Removing $(DESTDIR)$(PREFIX)/etc/rc.d/specter"
	@rm -f $(DESTDIR)$(PREFIX)/etc/rc.d/specter

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
	@echo "Left $(SYSCONFDIR)/specter.toml in place; run uninstall-config to remove it."
