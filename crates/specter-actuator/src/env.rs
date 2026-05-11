//! Captured operator environment for `${env.<NAME>}` resolution.
//!
//! `EnvSnapshot::capture` walks `std::env::vars_os` once at actuator
//! startup and freezes the result into a sorted map. The snapshot lives
//! on [`crate::pool::state::ActuatorState`] as `Arc<EnvSnapshot>` and is
//! handed to the resolver alongside every per-step resolve call.
//!
//! # Why a snapshot, not live reads
//!
//! Two reasons:
//!
//! 1. **Determinism (scoped).** Two spawn-time resolves of the same
//!    `${env.X}` placeholder, within one Effect's plan, must return
//!    the same value — even if a separate thread (or, in theory, the
//!    operator) mutates the process env between steps. A snapshot
//!    pins "what env did Specter start under" as the authoritative
//!    answer for **specter-mediated placeholder reads**. It is *not*
//!    a guarantee about what the child process sees when it reads env
//!    directly — see [`crate::os`]'s `build_command` for the additive
//!    parent-env contract; a child shell reading `$HOME` reads the
//!    daemon's live env, not the snapshot.
//! 2. **Cheap reads.** `std::env::vars_os` allocates and re-walks the
//!    OS env block on every call; per-step resolves would re-pay that
//!    cost. The snapshot is read with a `BTreeMap::get` plus a string
//!    borrow — no allocation on the lookup path.
//!
//! Specter never `setenv`s internally, so the snapshot doesn't go stale
//! under our own code. An operator wanting live re-reads on SIGHUP would
//! swap this for a `dyn EnvSource` (assumption A4 in the action-types
//! expansion plan).
//!
//! Non-UTF-8 entries are silently dropped at capture time. The lexer
//! [`crate::spawner`]'s upstream grammar guarantees placeholder names
//! are ASCII (`[A-Za-z_][A-Za-z0-9_]*`), and the resolver compares the
//! placeholder name byte-for-byte against the captured key — UTF-8 lossy
//! keys would never match a well-formed placeholder, so dropping them
//! costs nothing.

use compact_str::CompactString;
use std::collections::BTreeMap;
use std::ffi::OsString;

/// Frozen snapshot of the operator environment at actuator startup.
///
/// Backed by [`BTreeMap<CompactString, CompactString>`] — alphabetical
/// iteration is incidentally useful for diagnostics; the hot path is
/// [`Self::get`].
#[derive(Debug)]
pub(crate) struct EnvSnapshot {
    map: BTreeMap<CompactString, CompactString>,
}

impl EnvSnapshot {
    /// Capture the current process environment. Called once per
    /// actuator at startup. Non-UTF-8 keys or values are skipped.
    #[must_use]
    pub fn capture() -> Self {
        Self::from_vars_os(std::env::vars_os())
    }

    /// Build a snapshot from any iterator of `(OsString, OsString)`
    /// pairs — the same shape `std::env::vars_os` yields. Non-UTF-8
    /// keys or values are dropped silently.
    ///
    /// Production [`Self::capture`] delegates here; tests reach for
    /// this directly when they need to exercise the UTF-8 filter
    /// deterministically (without touching the ambient process env,
    /// which would require `unsafe std::env::set_var` and is racy
    /// across single-process test runners). [`Self::from_map`] is
    /// the lighter test fixture for the common case where the test
    /// doesn't care about the filter and only needs ASCII keys.
    #[must_use]
    fn from_vars_os<I>(vars: I) -> Self
    where
        I: IntoIterator<Item = (OsString, OsString)>,
    {
        let mut map = BTreeMap::new();
        for (k, v) in vars {
            if let (Ok(k), Ok(v)) = (k.into_string(), v.into_string()) {
                map.insert(CompactString::from(k), CompactString::from(v));
            }
        }
        Self { map }
    }

    /// Test fixture: build a snapshot from an iterator of pairs.
    ///
    /// Keys deduplicate via [`BTreeMap`]: a later pair with the same
    /// key wins, mirroring `std::env::set_var` semantics.
    ///
    /// Gated to `cfg(test)` because every call site lives inside
    /// `#[cfg(test)]` modules within this crate — exposing the
    /// fixture on a production build would warn `dead_code` without
    /// the gate. Tests below this module and in `pool::state::tests`
    /// see this via the same `cfg(test)` predicate.
    #[cfg(test)]
    #[must_use]
    pub fn from_map<I, K, V>(entries: I) -> Self
    where
        I: IntoIterator<Item = (K, V)>,
        K: Into<CompactString>,
        V: Into<CompactString>,
    {
        Self {
            map: entries
                .into_iter()
                .map(|(k, v)| (k.into(), v.into()))
                .collect(),
        }
    }

    /// Look up an env var by name. Returns `None` if the key is absent.
    #[must_use]
    pub fn get(&self, name: &str) -> Option<&str> {
        self.map.get(name).map(CompactString::as_str)
    }
}

#[cfg(test)]
mod tests {
    use super::EnvSnapshot;

    #[test]
    fn from_map_round_trips_entries() {
        let snap = EnvSnapshot::from_map([("HOME", "/home/op"), ("USER", "op")]);
        assert_eq!(snap.get("HOME"), Some("/home/op"));
        assert_eq!(snap.get("USER"), Some("op"));
        assert_eq!(snap.get("MISSING"), None);
    }

    #[test]
    fn from_map_later_overwrites_earlier_on_duplicate_key() {
        let snap = EnvSnapshot::from_map([("HOME", "/old"), ("HOME", "/new")]);
        assert_eq!(snap.get("HOME"), Some("/new"));
    }

    #[test]
    fn empty_snapshot_returns_none_for_any_key() {
        let snap = EnvSnapshot::from_map::<_, &str, &str>([]);
        assert!(snap.get("ANYTHING").is_none());
    }

    /// Exercise the production [`EnvSnapshot::from_vars_os`] pipeline
    /// (which [`EnvSnapshot::capture`] delegates to) on synthetic
    /// `(OsString, OsString)` pairs. Pins three behaviors at once:
    ///
    /// - UTF-8 keys and values round-trip through the snapshot.
    /// - Non-UTF-8 keys are silently dropped (matches the module
    ///   docstring: the lexer's grammar guarantees ASCII placeholder
    ///   names, so a non-UTF-8 key would never match a placeholder).
    /// - Non-UTF-8 values are silently dropped for the same reason —
    ///   the rendered argv slot would be replacement-char garbage
    ///   regardless.
    ///
    /// Avoids `unsafe std::env::set_var` (Rust 2024 marks it unsafe
    /// due to inherent data races against concurrent `getenv`) by
    /// driving the pipeline with synthetic input instead of the
    /// ambient process env.
    #[test]
    fn from_vars_os_round_trips_utf8_and_filters_non_utf8() {
        use std::ffi::OsString;
        use std::os::unix::ffi::OsStringExt as _;

        let invalid_key = OsString::from_vec(vec![0xff, 0xfe]);
        let invalid_val = OsString::from_vec(vec![0xff, 0xfe]);
        let snap = EnvSnapshot::from_vars_os([
            (OsString::from("HOME"), OsString::from("/home/op")),
            (OsString::from("USER"), OsString::from("op")),
            // Non-UTF-8 key — filtered.
            (invalid_key, OsString::from("filtered-by-key")),
            // Non-UTF-8 value — filtered.
            (OsString::from("BAD_VALUE"), invalid_val),
        ]);
        assert_eq!(snap.get("HOME"), Some("/home/op"));
        assert_eq!(snap.get("USER"), Some("op"));
        assert_eq!(snap.get("BAD_VALUE"), None, "non-UTF-8 value filtered");
        assert_eq!(snap.map.len(), 2, "only the two UTF-8 entries survive");
    }
}
