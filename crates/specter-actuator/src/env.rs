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
//! 1. **Determinism.** Two spawn-time resolves of the same `${env.X}`
//!    placeholder, within one Effect's plan, must return the same value
//!    — even if a separate thread (or, in theory, the operator) mutates
//!    the process env between steps. A snapshot pins "what env did
//!    Specter start under" as the authoritative answer.
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
        let mut map = BTreeMap::new();
        for (k, v) in std::env::vars_os() {
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

    #[test]
    fn capture_includes_a_known_env_var() {
        // Set a sentinel before capture so we don't depend on the
        // ambient environment (which may or may not have `PATH`,
        // `HOME`, etc. set under CI sandboxes).
        // SAFETY: this test is single-threaded; we set + read the
        // sentinel without racing other tests because nothing else
        // touches `SPECTER_ENVSNAPSHOT_SENTINEL`.
        #[allow(unsafe_code)]
        unsafe {
            std::env::set_var("SPECTER_ENVSNAPSHOT_SENTINEL", "captured");
        }
        let snap = EnvSnapshot::capture();
        assert_eq!(snap.get("SPECTER_ENVSNAPSHOT_SENTINEL"), Some("captured"));
        #[allow(unsafe_code)]
        unsafe {
            std::env::remove_var("SPECTER_ENVSNAPSHOT_SENTINEL");
        }
    }
}
