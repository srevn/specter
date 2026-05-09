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
    pub command: Vec<String>,
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
}

#[cfg(test)]
impl RawWatch {
    /// Construct a `RawWatch` directly for tests that exercise the
    /// validator helpers without routing through TOML deserialization
    /// (e.g., the `validate_static_watch` defensive `is_dynamic`
    /// re-check, which is unreachable through the dispatcher).
    pub(crate) fn for_test(name: String, path: String, command: Vec<String>) -> Self {
        Self {
            name,
            path,
            command,
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
        }
    }
}
