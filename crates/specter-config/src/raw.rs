use serde::Deserialize;
use std::time::Duration;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawConfig {
    /// Block of operator-facing engine telemetry settings — level, destination, file path. The
    /// schema groups them under one table: a top-level `log_level` is rejected as an unknown field
    /// (no migration). Use `[log]\nlevel = "debug"`.
    ///
    /// `#[serde(default)]` collapses "no `[log]` table" and "empty `[log]` table" into the same
    /// `RawLogConfig::default()` state (every field `None`). Both surface identically through
    /// `validate_log`, which unfolds the Nones to the [`crate::LogConfig`] defaults. Wrapping the
    /// field in an `Option<_>` would only duplicate that defaulting on the validator side without
    /// buying any extra distinction.
    #[serde(default)]
    pub log: RawLogConfig,
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
#[allow(clippy::struct_excessive_bools)]
pub(crate) struct RawWatch {
    pub name: String,
    pub path: String,
    /// Reaction body — sequence of [`RawAction`]s. Validation requires at least one entry; the
    /// actuator runs the steps sequentially with stop-on-failure.
    pub actions: Vec<RawAction>,
    /// Walk descendants of the anchor recursively. Default `true` — the recursive case is the
    /// dominant operator intent (a watch on `/srv/build` almost always wants files anywhere
    /// underneath). `false` confines the watch to the anchor's immediate children. A plain `bool`
    /// suffices: "absent" and "explicit `false`" carry no distinction, so the `default_true`
    /// default covers both.
    #[serde(default = "default_true")]
    pub recursive: bool,
    pub pattern: Option<String>,
    pub exclude: Option<Vec<String>>,
    /// Include dot-prefixed entries. Default `false` — dotfiles are editor swap files, VCS
    /// metadata, OS caches, almost never the signal the operator is watching. Set `hidden = true`
    /// for cases like `/home/$USER/.config` where the dotfiles ARE the payload.
    #[serde(default)]
    pub hidden: bool,
    /// Debounce window after the last event. TOML accepts humantime strings (`"200ms"`, `"1s"`,
    /// `"1m 30s"`); omitted ⇒ [`crate::config::DEFAULT_SETTLE`].
    #[serde(default, with = "humantime_serde")]
    pub settle: Option<Duration>,
    /// Forced-fire deadline after burst start. TOML accepts humantime strings (`"1h"`, `"30m"`);
    /// omitted ⇒ [`crate::config::DEFAULT_MAX_SETTLE`].
    #[serde(default, with = "humantime_serde")]
    pub max_settle: Option<Duration>,
    pub scope: Option<String>,
    pub max_depth: Option<u32>,
    pub events: Option<Vec<String>>,
    /// Forward subprocess stdout/stderr to Specter's own stdio. Default `false` — the engine
    /// threads `Stdio::null()` so a chatty hook doesn't flood the operator's console; setting this
    /// to `true` routes child output through the supervisor's log facility (systemd journal,
    /// launchd `StandardOutPath`).
    #[serde(default)]
    pub log_output: bool,
    /// Suppress this entry without removing it from the TOML. Default `true` — present entries are
    /// effective by default; `false` is structurally equivalent to "absent from the config"
    /// (filtered by [`crate::Config::active_watches`] / [`crate::Config::active_promoters`] before
    /// the engine sees anything).
    #[serde(default = "default_true")]
    pub enabled: bool,
}

const fn default_true() -> bool {
    true
}

/// One entry in `actions = [...]`. Variants light up additively as sibling `Option<…>` fields:
/// `exec` for a single process, `pipe` for a chain of processes wired stdout→stdin, and the `when`
/// / `then` / `else` triple for conditionals. `deny_unknown_fields` catches typos at the variant
/// tag level (e.g., `exce`, `paralel`).
///
/// Validation enforces "exactly one variant set" — the variant tags `exec`, `pipe`, and `when` are
/// mutually exclusive on a single `[[watch.actions]]` entry. Inside a conditional, `when` is the
/// predicate and `then` / `else` carry nested `RawAction` arrays (full recursive grammar). Inside a
/// pipe, each stage is a [`RawExec`] — `exec` argv plus an optional per-stage `timeout`.
#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawAction {
    /// `Some(argv)` for `{ exec = [...] }` actions. Mutually exclusive with `when` and `pipe`.
    pub exec: Option<Vec<String>>,
    /// `Some(stages)` for `{ pipe = [{ exec = [...] }, ...] }` actions. Each stage is one
    /// [`RawExec`] (its own argv + optional per-stage `timeout`); the actuator wires their stdouts
    /// to the next stage's stdin via `pipe(2)` and aggregates outcomes with pipefail-on semantics.
    /// Validation rejects empty arrays and single-stage pipes (use top-level `exec` directly).
    pub pipe: Option<Vec<RawExec>>,
    /// Predicate of a conditional action. `Some(predicate)` opens the `when` / `then` / `else`
    /// group; the validator requires `then` alongside it. The predicate carries its own per-step
    /// `timeout` inside the nested [`RawExec`].
    pub when: Option<RawExec>,
    /// Then-branch of a conditional action. Required alongside `when`; rejected when `when` is
    /// absent. The actuator runs the `then` body on predicate Ok; full `RawAction` grammar is
    /// recursive here (nested conditionals are allowed).
    pub then: Option<Vec<Self>>,
    /// Else-branch of a conditional action. Optional even when `when` is set: omitting `else` makes
    /// the predicate's Failed outcome skip past the conditional with no propagation. TOML field
    /// name is `else` (a Rust keyword); serde renames it to the Rust-side `otherwise` identifier.
    #[serde(default, rename = "else")]
    pub otherwise: Option<Vec<Self>>,
    /// Per-step deadline in humantime format (`"500ms"`, `"30s"`, `"5m"`). Valid only on
    /// `exec`-bearing actions; validation rejects it on non-exec variants. Pipe stages each set
    /// their own `timeout` on the nested [`RawExec`]; the conditional predicate sets it inside the
    /// `when` table. `None` ⇒ no deadline. Threaded onto [`specter_core::ExecAction::timeout`] and
    /// enforced by the actuator's per-step timer thread: SIGTERM at `now + timeout`, SIGKILL after
    /// the actuator's `shutdown_grace`.
    #[serde(default, with = "humantime_serde")]
    pub timeout: Option<Duration>,
}

/// Nested-exec table used inside the predicate slot of a conditional action (`when = { exec =
/// [...], timeout = "5s" }`) and inside each stage of a `pipe = [...]` action.
///
/// The shape is intentionally tighter than [`RawAction`]: only the `exec` and `timeout` fields are
/// allowed, no variant tags. A predicate cannot itself be a pipe or a nested conditional, and a
/// pipe stage cannot itself be a pipe or a conditional. A future relaxation would swap this for a
/// `Box<RawAction>` and let the validator recurse.
///
/// `Deserialize` is hand-rolled (rather than derived with `deny_unknown_fields`) so the common
/// operator mistake of putting a `pipe`/`when`/`then`/`else` tag inside a predicate or pipe stage
/// surfaces with a hint pointing at the top-level form, instead of an opaque "unknown field" message.
#[derive(Debug)]
pub(crate) struct RawExec {
    pub exec: Vec<String>,
    pub timeout: Option<Duration>,
}

impl<'de> serde::Deserialize<'de> for RawExec {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        deserializer.deserialize_map(RawExecVisitor)
    }
}

struct RawExecVisitor;

impl<'de> serde::de::Visitor<'de> for RawExecVisitor {
    type Value = RawExec;

    fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
        formatter.write_str("a table with `exec` (and optional `timeout`)")
    }

    fn visit_map<A: serde::de::MapAccess<'de>>(self, mut map: A) -> Result<RawExec, A::Error> {
        let mut exec: Option<Vec<String>> = None;
        let mut timeout: Option<Duration> = None;

        while let Some(key) = map.next_key::<String>()? {
            match key.as_str() {
                "exec" => {
                    if exec.is_some() {
                        return Err(serde::de::Error::duplicate_field("exec"));
                    }
                    exec = Some(map.next_value()?);
                }
                "timeout" => {
                    if timeout.is_some() {
                        return Err(serde::de::Error::duplicate_field("timeout"));
                    }
                    // Wrapper adapts humantime_serde::deserialize (which takes a Deserializer) to
                    // MapAccess::next_value (which takes a Deserialize impl).
                    timeout = Some(map.next_value::<HumantimeDuration>()?.0);
                }
                // Variant tags valid at the top-level `[[watch.actions]]` row but rejected inside a
                // nested exec slot (predicate `when`, pipe stage). The `deny_unknown_fields` default
                // would surface this as "unknown field `pipe`, expected one of `exec`, `timeout`" —
                // true but unhelpful. Emit a hint that names the structural rule instead.
                "pipe" | "when" | "then" | "else" => {
                    return Err(serde::de::Error::custom(format!(
                        "`{key}` not allowed inside a nested exec table — only \
                         `exec` and `timeout` are valid here. Conditionals and \
                         pipes must appear at the top level of `[[watch.actions]]`."
                    )));
                }
                _ => {
                    return Err(serde::de::Error::unknown_field(&key, &["exec", "timeout"]));
                }
            }
        }

        let exec = exec.ok_or_else(|| serde::de::Error::missing_field("exec"))?;
        Ok(RawExec { exec, timeout })
    }
}

/// Adapter that routes `Duration` through `humantime_serde` for use inside
/// `MapAccess::next_value::<T>()` calls where `T: Deserialize`. Equivalent to `#[serde(with =
/// "humantime_serde")]` on a derived field, but usable from the hand-rolled visitor above.
struct HumantimeDuration(Duration);

impl<'de> serde::Deserialize<'de> for HumantimeDuration {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        humantime_serde::deserialize(deserializer).map(HumantimeDuration)
    }
}
