//! Subscription, command templates, and `EffectScope`.
//!
//! `Sub.needs_diff` is derived at construction: true iff the `EffectScope`
//! is `PerStableFile` *or* the command template references any diff-derived
//! placeholder (`Created`/`Deleted`/`Modified`/`RenamedFrom`/`RenamedTo`).
//!
//! v1 surface is argv-only — no shell variant.

use crate::ids::{ProfileId, ResourceId, SubId};
use crate::scan_config::ScanConfig;
use compact_str::CompactString;
use slotmap::SlotMap;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::time::Duration;
use tinyvec::TinyVec;

/// Public-API request to attach a Sub.
///
/// Carries everything `Engine::attach_sub` needs to either reuse an
/// existing Profile (matching `(resource, config_hash)`) or create a fresh
/// one. Two ways to identify the anchor:
///
/// - `resource: ResourceId` for paths the engine has already materialized
///   (P4 path; bin/test code that walks the Tree first).
/// - `path: Some(PathBuf)` for absolute paths the engine should
///   materialize itself (pending-path support).
///   When `path` is `Some`, the engine ignores `resource` and walks the
///   path components via `Tree::ensure_path`. If any non-leaf component is
///   a fresh `DescentScaffold`, the Profile is registered as pending; once
///   the anchor materializes, a `Burst { intent: Seed }` establishes the
///   baseline.
///
/// `name` is `String` so callers don't need a `compact_str` dependency at
/// this seam — `Sub::new` converts via `Into<CompactString>` internally.
///
/// Lives in `core::sub` rather than `engine::engine` so
/// [`SubRegistryDiff`] (a `core` type, consumed via
/// [`Input::ConfigDiff`]) can carry pre-id `SubAttachRequest`s without
/// introducing a `core → engine` cycle. `Clone` is derived for the
/// (rare) call sites that fan a request out to multiple Engines —
/// production paths consume by value.
#[derive(Clone, Debug)]
pub struct SubAttachRequest {
    pub name: String,
    pub resource: ResourceId,
    pub path: Option<PathBuf>,
    pub config: ScanConfig,
    pub max_settle: Duration,
    pub settle: Duration,
    pub command: CommandTemplate,
    pub scope: EffectScope,
}

impl SubAttachRequest {
    /// Build a request anchored at a pre-materialized `ResourceId`. The
    /// engine looks up the Resource by id; the request does not carry a
    /// path string.
    #[must_use]
    pub const fn for_resource(
        name: String,
        resource: ResourceId,
        config: ScanConfig,
        max_settle: Duration,
        settle: Duration,
        command: CommandTemplate,
        scope: EffectScope,
    ) -> Self {
        Self {
            name,
            resource,
            path: None,
            config,
            max_settle,
            settle,
            command,
            scope,
        }
    }

    /// Build a request anchored at an absolute path. The engine walks the
    /// path via `Tree::ensure_path` at attach time and decides between
    /// immediate Seed and pending descent based on which segments already
    /// exist. `resource` defaults to `ResourceId::default()`; the engine
    /// treats that as the "use the path" signal.
    #[must_use]
    pub fn for_path(
        name: String,
        path: PathBuf,
        config: ScanConfig,
        max_settle: Duration,
        settle: Duration,
        command: CommandTemplate,
        scope: EffectScope,
    ) -> Self {
        Self {
            name,
            resource: ResourceId::default(),
            path: Some(path),
            config,
            max_settle,
            settle,
            command,
            scope,
        }
    }
}

/// Hot-reload diff. Computed by the TOML loader; consumed by
/// `Engine::step(Input::ConfigDiff(_))`.
///
/// `added` and `modified` carry pre-id [`SubAttachRequest`] data; `removed`
/// carries existing [`SubId`]s. Engine processes `removed → modified →
/// added` atomically in one step, with parent-edge recompute after each
/// detach/attach.
#[derive(Clone, Debug, Default)]
pub struct SubRegistryDiff {
    pub added: Vec<SubAttachRequest>,
    pub removed: Vec<SubId>,
    pub modified: Vec<(SubId, SubAttachRequest)>,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum EffectScope {
    #[default]
    SubtreeRoot,
    PerStableFile,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CommandTemplate {
    pub argv: Vec<ArgTemplate>,
}

impl CommandTemplate {
    #[must_use]
    pub fn new(argv: impl IntoIterator<Item = ArgTemplate>) -> Self {
        Self {
            argv: argv.into_iter().collect(),
        }
    }

    /// `true` iff any argv part references a diff-derived placeholder
    /// (`$created`/`$deleted`/`$modified`/`$renamed`-from-or-to). Computed
    /// once at `Sub::new`; never re-evaluated.
    #[must_use]
    pub fn references_diff(&self) -> bool {
        self.argv
            .iter()
            .any(|arg| arg.parts.iter().any(ArgPart::is_diff_placeholder))
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ArgTemplate {
    pub parts: SmallVec<[ArgPart; 2]>,
}

impl ArgTemplate {
    #[must_use]
    pub fn new(parts: impl IntoIterator<Item = ArgPart>) -> Self {
        Self {
            parts: parts.into_iter().collect(),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ArgPart {
    Literal(CompactString),
    Placeholder(Placeholder),
}

impl ArgPart {
    #[must_use]
    pub fn literal(s: impl Into<CompactString>) -> Self {
        Self::Literal(s.into())
    }

    #[must_use]
    pub const fn is_diff_placeholder(&self) -> bool {
        use Placeholder::{Created, Deleted, Modified, RenamedFrom, RenamedTo};
        matches!(
            self,
            Self::Placeholder(Created | Deleted | Modified | RenamedFrom | RenamedTo)
        )
    }
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum Placeholder {
    Path,
    Rel,
    Anchor,
    Created,
    Deleted,
    Modified,
    RenamedFrom,
    RenamedTo,
}

#[derive(Debug)]
pub struct Sub {
    pub id: SubId,
    pub name: CompactString,
    pub profile: ProfileId,
    pub command: CommandTemplate,
    pub scope: EffectScope,
    pub settle: Duration,
    pub max_settle: Duration,
    pub needs_diff: bool,
}

impl Sub {
    /// Construct a Sub. `needs_diff` is derived: true iff
    /// `scope == PerStableFile` OR the template references any diff entry.
    /// Pre-computed once; never re-evaluated.
    #[must_use]
    pub fn new(
        id: SubId,
        name: impl Into<CompactString>,
        profile: ProfileId,
        command: CommandTemplate,
        scope: EffectScope,
        settle: Duration,
        max_settle: Duration,
    ) -> Self {
        let needs_diff = scope == EffectScope::PerStableFile || command.references_diff();
        Self {
            id,
            name: name.into(),
            profile,
            command,
            scope,
            settle,
            max_settle,
            needs_diff,
        }
    }
}

#[derive(Debug, Default)]
pub struct SubRegistry {
    subs: SlotMap<SubId, Sub>,
    by_profile: BTreeMap<ProfileId, TinyVec<[SubId; 2]>>,
}

impl SubRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a Sub built from the freshly-minted `SubId`. The Sub stores
    /// its own id; the closure embeds the minted key into the Sub.
    pub fn insert<F>(&mut self, build: F) -> SubId
    where
        F: FnOnce(SubId) -> Sub,
    {
        let id = self.subs.insert_with_key(build);
        let profile = self.subs[id].profile;
        self.by_profile.entry(profile).or_default().push(id);
        id
    }

    pub fn remove(&mut self, id: SubId) -> Option<Sub> {
        let sub = self.subs.remove(id)?;
        if let Some(v) = self.by_profile.get_mut(&sub.profile) {
            v.retain(|sid| *sid != id);
            if v.is_empty() {
                self.by_profile.remove(&sub.profile);
            }
        }
        Some(sub)
    }

    #[must_use]
    pub fn get(&self, id: SubId) -> Option<&Sub> {
        self.subs.get(id)
    }

    /// Subs attached to `profile`, in insertion order. Empty slice if none.
    #[must_use]
    pub fn at(&self, profile: ProfileId) -> &[SubId] {
        self.by_profile.get(&profile).map_or(&[], |v| v.as_slice())
    }

    pub fn iter(&self) -> impl Iterator<Item = (SubId, &Sub)> {
        self.subs.iter()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.subs.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.subs.is_empty()
    }

    /// Linear-scan lookup of a Sub by its user-facing `name`. `O(N_subs)`
    /// per call; uniqueness is the caller's responsibility (the loader's
    /// validation rejects duplicates upstream — when two Subs share a
    /// name the first match in `SlotMap` iteration order wins).
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<SubId> {
        self.subs
            .iter()
            .find_map(|(id, s)| (s.name == name).then_some(id))
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ArgPart, ArgTemplate, CommandTemplate, EffectScope, Placeholder, Sub, SubRegistry,
    };
    use crate::ids::{ProfileId, SubId};
    use std::time::Duration;

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);

    fn anchor_only_template() -> CommandTemplate {
        CommandTemplate::new([ArgTemplate::new([
            ArgPart::literal("/bin/build"),
            ArgPart::Placeholder(Placeholder::Path),
        ])])
    }

    fn template_with(p: Placeholder) -> CommandTemplate {
        CommandTemplate::new([ArgTemplate::new([ArgPart::Placeholder(p)])])
    }

    #[test]
    fn references_diff_for_each_diff_placeholder() {
        for p in [
            Placeholder::Created,
            Placeholder::Deleted,
            Placeholder::Modified,
            Placeholder::RenamedFrom,
            Placeholder::RenamedTo,
        ] {
            assert!(
                template_with(p).references_diff(),
                "references_diff must be true for {p:?}"
            );
        }
    }

    #[test]
    fn references_diff_false_for_anchor_only_template() {
        assert!(!anchor_only_template().references_diff());
        for p in [Placeholder::Path, Placeholder::Rel, Placeholder::Anchor] {
            assert!(
                !template_with(p).references_diff(),
                "references_diff must be false for anchor-only {p:?}"
            );
        }
    }

    #[test]
    fn needs_diff_set_for_per_stable_file_scope() {
        let sub = Sub::new(
            SubId::default(),
            "fmt",
            ProfileId::default(),
            anchor_only_template(),
            EffectScope::PerStableFile,
            SETTLE,
            MAX_SETTLE,
        );
        assert!(sub.needs_diff);
    }

    #[test]
    fn needs_diff_set_for_diff_placeholder_in_subtree_scope() {
        let sub = Sub::new(
            SubId::default(),
            "report",
            ProfileId::default(),
            template_with(Placeholder::Created),
            EffectScope::SubtreeRoot,
            SETTLE,
            MAX_SETTLE,
        );
        assert!(sub.needs_diff);
    }

    #[test]
    fn needs_diff_false_for_anchor_subtree_combo() {
        let sub = Sub::new(
            SubId::default(),
            "build",
            ProfileId::default(),
            anchor_only_template(),
            EffectScope::SubtreeRoot,
            SETTLE,
            MAX_SETTLE,
        );
        assert!(!sub.needs_diff);
    }

    #[test]
    fn registry_insert_embeds_minted_id() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(|id| {
            Sub::new(
                id,
                "build",
                pid,
                anchor_only_template(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
            )
        });

        let sub = reg.get(sid).expect("Sub stored");
        assert_eq!(sub.id, sid, "Sub.id matches the minted key");
        assert_eq!(sub.name, "build");
    }

    #[test]
    fn registry_at_groups_by_profile() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();

        let s1 = reg.insert(|id| {
            Sub::new(
                id,
                "a",
                pid,
                anchor_only_template(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
            )
        });
        let s2 = reg.insert(|id| {
            Sub::new(
                id,
                "b",
                pid,
                anchor_only_template(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
            )
        });

        let mut got = reg.at(pid).to_vec();
        got.sort();
        let mut expected = vec![s1, s2];
        expected.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn registry_at_empty_for_unknown_profile() {
        let reg = SubRegistry::new();
        assert!(reg.at(ProfileId::default()).is_empty());
    }

    #[test]
    fn find_by_name_returns_some_for_match() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let id = reg.insert(|id| {
            Sub::new(
                id,
                "build",
                pid,
                anchor_only_template(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
            )
        });
        assert_eq!(reg.find_by_name("build"), Some(id));
    }

    #[test]
    fn find_by_name_returns_none_for_absent() {
        let reg = SubRegistry::new();
        assert!(reg.find_by_name("missing").is_none());
    }

    #[test]
    fn find_by_name_returns_none_after_remove() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let id = reg.insert(|id| {
            Sub::new(
                id,
                "build",
                pid,
                anchor_only_template(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
            )
        });
        reg.remove(id);
        assert!(reg.find_by_name("build").is_none());
    }

    #[test]
    fn find_by_name_resolves_one_of_duplicates() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let a = reg.insert(|id| {
            Sub::new(
                id,
                "shared",
                pid,
                anchor_only_template(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
            )
        });
        let b = reg.insert(|id| {
            Sub::new(
                id,
                "shared",
                pid,
                anchor_only_template(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
            )
        });
        let found = reg.find_by_name("shared").expect("at least one match");
        assert!(found == a || found == b, "find returns one of the matches");
    }

    #[test]
    fn registry_remove_clears_by_profile_and_drops_empty_bucket() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(|id| {
            Sub::new(
                id,
                "build",
                pid,
                anchor_only_template(),
                EffectScope::SubtreeRoot,
                SETTLE,
                MAX_SETTLE,
            )
        });

        let removed = reg.remove(sid);
        assert!(removed.is_some());
        assert!(reg.get(sid).is_none());
        assert!(reg.at(pid).is_empty());
        assert_eq!(reg.len(), 0);
    }
}
