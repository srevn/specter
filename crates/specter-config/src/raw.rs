use serde::Deserialize;
use std::time::Duration;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawConfig {
    /// Block of operator-facing engine telemetry settings — level,
    /// destination, file path. v1 splits the schema cleanly: the top-level
    /// `log_level` of older configs no longer parses (alpha break, no
    /// migration). Use `[log]\nlevel = "debug"`.
    #[serde(default)]
    pub log: Option<RawLogConfig>,
    #[serde(default, rename = "watch")]
    pub watches: Vec<RawWatch>,
}

#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawLogConfig {
    pub level: Option<String>,
    pub destination: Option<String>,
    pub path: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawWatch {
    pub name: String,
    pub path: String,
    /// Reaction body — sequence of [`RawAction`]s. Replaces the v0
    /// `command: Vec<String>` field. Validation requires at least one
    /// entry; the actuator runs the steps sequentially with
    /// stop-on-failure.
    pub actions: Vec<RawAction>,
    pub recursive: Option<bool>,
    pub pattern: Option<String>,
    pub exclude: Option<Vec<String>>,
    pub hidden: Option<bool>,
    /// Debounce window after the last event. TOML accepts humantime
    /// strings (`"200ms"`, `"1s"`, `"1m 30s"`); omitted ⇒
    /// [`crate::config::DEFAULT_SETTLE`].
    #[serde(default, with = "humantime_serde")]
    pub settle: Option<Duration>,
    /// Forced-fire deadline after burst start. TOML accepts humantime
    /// strings (`"1h"`, `"30m"`); omitted ⇒
    /// [`crate::config::DEFAULT_MAX_SETTLE`].
    #[serde(default, with = "humantime_serde")]
    pub max_settle: Option<Duration>,
    pub scope: Option<String>,
    pub max_depth: Option<u32>,
    pub events: Option<Vec<String>>,
    pub log_output: Option<bool>,
    pub enabled: Option<bool>,
}

/// One entry in `actions = [...]`. Variants light up additively as
/// sibling `Option<…>` fields: `exec` for a single process, the
/// `when` / `then` / `else` triple for conditionals, and (future)
/// `pipe` for piped stages. `deny_unknown_fields` catches typos at
/// the variant tag level (e.g., `exce`, `paralel`).
///
/// Validation enforces "exactly one variant set" — the variant tags
/// `exec`, `when` and (future) `pipe` are mutually exclusive on a
/// single `[[watch.actions]]` entry. Inside a conditional, `when` is
/// the predicate and `then` / `else` carry nested `RawAction` arrays
/// (full recursive grammar).
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawAction {
    /// `Some(argv)` for `{ exec = [...] }` actions. Mutually exclusive
    /// with `when` (and future `pipe`).
    pub exec: Option<Vec<String>>,
    /// Predicate of a conditional action. `Some(predicate)` opens the
    /// `when` / `then` / `else` group; the validator requires `then`
    /// alongside it. The predicate carries its own per-step `timeout`
    /// inside the nested [`RawExec`].
    pub when: Option<RawExec>,
    /// Then-branch of a conditional action. Required alongside
    /// `when`; rejected when `when` is absent. The actuator runs the
    /// `then` body on predicate Ok; full `RawAction` grammar is
    /// recursive here (nested conditionals are allowed).
    pub then: Option<Vec<Self>>,
    /// Else-branch of a conditional action. Optional even when `when`
    /// is set: omitting `else` makes the predicate's Failed outcome
    /// skip past the conditional with no propagation. TOML field name
    /// is `else` (a Rust keyword); serde renames it to the Rust-side
    /// `otherwise` identifier.
    #[serde(default, rename = "else")]
    pub otherwise: Option<Vec<Self>>,
    /// Per-step deadline in humantime format (`"500ms"`, `"30s"`,
    /// `"5m"`). Valid only on `exec`-bearing actions; validation rejects
    /// it on non-exec variants (predicates carry their own per-step
    /// `timeout` inside the `when` table; future `pipe` stages will
    /// each carry their own). `None` ⇒ no deadline. Threaded onto
    /// [`specter_core::ExecAction::timeout`] and enforced by the
    /// actuator's per-step timer thread: SIGTERM at `now + timeout`,
    /// SIGKILL after the actuator's `shutdown_grace`.
    #[serde(default, with = "humantime_serde")]
    pub timeout: Option<Duration>,
}

/// Nested-exec table used inside the predicate slot of a conditional
/// action (`when = { exec = [...], timeout = "5s" }`). Future `pipe`
/// stages will reuse this same shape — one `RawExec` per stage.
///
/// The shape is intentionally tighter than [`RawAction`]: only the
/// `exec` and `timeout` fields are allowed, no variant tags. A
/// predicate cannot itself be a pipe or a nested conditional. A
/// future relaxation would swap this for a `Box<RawAction>` and let
/// the validator recurse.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawExec {
    pub exec: Vec<String>,
    #[serde(default, with = "humantime_serde")]
    pub timeout: Option<Duration>,
}

#[cfg(test)]
impl RawWatch {
    /// Construct a `RawWatch` directly for tests that exercise the
    /// validator helpers without routing through TOML deserialization
    /// (e.g., the `validate_static_watch` defensive `is_dynamic`
    /// re-check, which is unreachable through the dispatcher).
    pub(crate) fn for_test(name: String, path: String, exec: Vec<String>) -> Self {
        Self {
            name,
            path,
            actions: vec![RawAction {
                exec: Some(exec),
                when: None,
                then: None,
                otherwise: None,
                timeout: None,
            }],
            recursive: None,
            pattern: None,
            exclude: None,
            hidden: None,
            settle: None,
            max_settle: None,
            scope: None,
            max_depth: None,
            events: None,
            log_output: None,
            enabled: None,
        }
    }
}
