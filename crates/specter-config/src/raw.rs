use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RawConfig {
    pub log_level: Option<String>,
    #[serde(default, rename = "watch")]
    pub watches: Vec<RawWatch>,
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
}
