use std::fmt;
use std::path::{Path, PathBuf};

#[non_exhaustive]
#[derive(Debug)]
pub enum ConfigError {
    Io {
        path: PathBuf,
        source: std::io::Error,
    },
    Parse {
        path: Option<PathBuf>,
        message: String,
    },
    Validate {
        path: Option<PathBuf>,
        errors: Vec<ValidationIssue>,
    },
}

#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ValidationIssue {
    pub watch_index: Option<usize>,
    pub field: &'static str,
    pub kind: IssueKind,
    pub detail: String,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
pub enum IssueKind {
    Empty,
    EmptyCommand,
    EmptyArgv,
    NonAbsolute,
    NotCanonical,
    InvalidGlob,
    UnknownPlaceholder,
    SettleTooSmall,
    MaxSettleTooSmall,
    MaxSettleTooLarge,
    MaxDepthZero,
    DuplicateName,
    InvalidEnum,
}

impl ConfigError {
    pub(crate) fn parse(path: Option<&Path>, e: &toml::de::Error) -> Self {
        Self::Parse {
            path: path.map(Path::to_owned),
            message: e.to_string(),
        }
    }
}

impl fmt::Display for ConfigError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => {
                write!(f, "io error reading {}: {source}", path.display())
            }
            Self::Parse { path, message } => match path {
                Some(p) => write!(f, "parse error in {}: {message}", p.display()),
                None => write!(f, "parse error: {message}"),
            },
            Self::Validate { path, errors } => {
                let prefix = path
                    .as_ref()
                    .map_or_else(|| "<inline>".to_owned(), |p| p.display().to_string());
                writeln!(f, "{prefix}: {} validation error(s):", errors.len())?;
                for e in errors {
                    writeln!(f, "  {e}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ConfigError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        if let Self::Io { source, .. } = self {
            Some(source)
        } else {
            None
        }
    }
}

impl ValidationIssue {
    pub(crate) const fn new(
        watch_index: Option<usize>,
        field: &'static str,
        kind: IssueKind,
        detail: String,
    ) -> Self {
        Self {
            watch_index,
            field,
            kind,
            detail,
        }
    }
}

impl fmt::Display for ValidationIssue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self.watch_index {
            Some(i) => write!(
                f,
                "watch[{i}].{}: {} ({})",
                self.field,
                self.detail,
                kind_label(self.kind)
            ),
            None => write!(
                f,
                "{}: {} ({})",
                self.field,
                self.detail,
                kind_label(self.kind)
            ),
        }
    }
}

const fn kind_label(k: IssueKind) -> &'static str {
    match k {
        IssueKind::Empty => "empty",
        IssueKind::EmptyCommand => "empty-command",
        IssueKind::EmptyArgv => "empty-argv",
        IssueKind::NonAbsolute => "non-absolute",
        IssueKind::NotCanonical => "not-canonical",
        IssueKind::InvalidGlob => "invalid-glob",
        IssueKind::UnknownPlaceholder => "unknown-placeholder",
        IssueKind::SettleTooSmall => "settle-too-small",
        IssueKind::MaxSettleTooSmall => "max-settle-too-small",
        IssueKind::MaxSettleTooLarge => "max-settle-too-large",
        IssueKind::MaxDepthZero => "max-depth-zero",
        IssueKind::DuplicateName => "duplicate-name",
        IssueKind::InvalidEnum => "invalid-enum",
    }
}
