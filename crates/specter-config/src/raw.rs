use serde::Deserialize;

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
    pub settle_ms: Option<u64>,
    pub max_settle_ms: Option<u64>,
    pub scope: Option<String>,
    pub max_depth: Option<u32>,
    pub events: Option<Vec<String>>,
    pub log_output: Option<bool>,
}
