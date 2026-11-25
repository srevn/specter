//! The probe walkers, one per [`specter_core::ProbeRequest`] variant: [`probe_anchor_file`] and
//! [`probe_descent`] each `lstat` one path; [`probe_subtree`] walks a directory and certifies the
//! walk.
//!
//! A walk reads a directory's level whole — the open, every entry, every child's `lstat` — before
//! it enters any child directory, and it classifies a failed read in one place
//! ([`WalkContext::level_fault`]), as one of three things said of the level:
//!
//! - **Gone** (`ENOENT`, `ENOTDIR`, `ELOOP`: no longer a directory): left out of its parent's
//!   listing, and `Vanished` at the walk's root. A child's own `ENOENT` leaves the level whole.
//! - **Want of a resource**: it says nothing of the tree, so the walk stops with `Failed(Transient)`.
//! - **Denied**, anything else: the level cannot be read whole, so it is denied as one, and what
//!   stands in for it is decided by [`WalkContext::stand_in`] — its earlier listing, matched by
//!   name and identity, or [`DirChild::Denied`], an identity with no content. A root with no
//!   earlier listing is one the walk cannot answer for: `Failed(Denied)`.
//!
//! The certificate names every directory the walk had cause to read and could not. A subtree
//! watch's lever over a directory it may not read is `exclude`, whose gate runs before the `lstat`.
//!
//! An unforced walk does not read a directory it owes no read — no obligation runs through it and
//! its `lstat` has not moved since its earlier listing — and stands on that listing, at every
//! depth. Symlinks are never followed.

use crate::ProbeFailureExt;
use crate::prober::PROBER_FAULT;
use compact_str::CompactString;
use specter_core::{
    ChildEntry, DirChild, DirMeta, DirSnapshot, EntryKind, FsIdentity, LeafEntry, PriorDir,
    ProbeFailure, ProbeOutcome, ProofAuthority, ProofObligation, ScanConfig,
};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsStr;
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;

/// What a walk can say of one directory it reached. `T` is what a listing carries: a level's own
/// entries, or the finished snapshot.
enum Reached<T> {
    Listed(T),
    Gone,
    Denied { errno: i32 },
}

impl Reached<Arc<DirSnapshot>> {
    /// What the parent's listing holds for the child directory of identity `fs_id`: nothing, once
    /// it is gone.
    fn into_dir_child(self, fs_id: FsIdentity) -> Option<DirChild> {
        match self {
            Self::Listed(snapshot) => Some(DirChild::Covered(snapshot)),
            Self::Gone => None,
            Self::Denied { .. } => Some(DirChild::Denied(fs_id)),
        }
    }
}

/// One level, read whole: what its listing holds without entering another directory.
#[derive(Default)]
struct Level {
    /// The leaves, and the boundary directories the shape does not enter.
    entries: BTreeMap<CompactString, ChildEntry>,
    /// The directories the shape enters, once the level is read whole.
    subdirs: Vec<Subdir>,
}

/// A directory a level holds that the walk enters, with the `lstat` that named it.
struct Subdir {
    name: CompactString,
    path: PathBuf,
    meta: DirMeta,
}

/// The inputs every directory of one walk shares, built once by [`probe_subtree`].
#[derive(Clone, Copy)]
struct WalkContext<'a> {
    /// The scope basis: every dirent's depth and gates are measured from the Profile anchor, as the
    /// engine's `covers` measures them, even when the walk roots below it.
    anchor_path: &'a Path,
    config: &'a ScanConfig,
    /// Stamped on every listing, so that two Profiles sharing a directory under different shapes
    /// never share a `dir_hash`.
    captured_with: u64,
    obligation: &'a ProofObligation,
    forced: bool,
    /// The walk root's device, which a subtree watch does not leave.
    root_dev: u64,
}

impl WalkContext<'_> {
    /// Whether the walk enters a child directory at `depth` on device `dev`: the shape's recursion
    /// edge, told what core cannot see — whether `dev` is the walk root's.
    fn should_recurse(&self, depth: u32, dev: u64) -> bool {
        self.config.descends_into(depth, dev == self.root_dev)
    }

    /// True iff an obligation chain runs at-or-through `path`.
    fn chain_through(&self, path: &Path) -> bool {
        match self.obligation {
            ProofObligation::Chains(chains) => chains.any_at_or_under(path),
            ProofObligation::WholeSubtree => false,
        }
    }

    /// True iff the walk has cause to read the directory at `path` afresh over `listing`, its
    /// earlier listing: an obligation runs through it, or its `lstat` (`root_meta`) moved since.
    fn owes_read(&self, path: &Path, root_meta: DirMeta, listing: &DirSnapshot) -> bool {
        matches!(self.obligation, ProofObligation::WholeSubtree)
            || listing.root_meta() != root_meta
            || self.chain_through(path)
    }

    /// The earlier listing an unforced walk stands on without reading the directory.
    fn mtime_skip<'p>(
        &self,
        path: &Path,
        root_meta: DirMeta,
        prior: PriorDir<'p>,
    ) -> Option<&'p Arc<DirSnapshot>> {
        if self.forced {
            return None;
        }
        prior
            .listing()
            .filter(|listing| !self.owes_read(path, root_meta, listing))
    }

    /// What stands in for the directory at `path`, which the walk could not read whole, and whether
    /// `denied` names it. Forcing a walk makes it read a directory, but gives it no cause to name
    /// one it owed no read.
    fn stand_in(
        &self,
        path: &Path,
        root_meta: DirMeta,
        prior: PriorDir<'_>,
        errno: i32,
        denied: &mut BTreeSet<Arc<Path>>,
    ) -> Reached<Arc<DirSnapshot>> {
        match prior {
            PriorDir::Listed(listing) => {
                if self.owes_read(path, root_meta, listing) {
                    denied.insert(Arc::from(path));
                }
                Reached::Listed(Arc::clone(listing))
            }
            // The earlier snapshot already holds it denied.
            PriorDir::Denied => Reached::Denied { errno },
            PriorDir::Unseen => {
                denied.insert(Arc::from(path));
                Reached::Denied { errno }
            }
        }
    }

    /// What a failed read says of the level it was reading — the one place a walk classifies an
    /// error. `read` is the path the read named: the level, or one of its children.
    fn level_fault(
        &self,
        read: &Path,
        e: &io::Error,
        prior: PriorDir<'_>,
    ) -> Result<Reached<Level>, ProbeFailure> {
        let anchor = self.anchor_path;
        if matches!(
            e.raw_os_error(),
            Some(libc::ENOENT | libc::ENOTDIR | libc::ELOOP)
        ) {
            tracing::trace!(?anchor, ?read, ?e, "probe_subtree: the level is gone");
            return Ok(Reached::Gone);
        }
        match ProbeFailure::from_io(e) {
            failure @ ProbeFailure::Transient { .. } => {
                tracing::debug!(?anchor, ?read, ?e, "probe_subtree: want of a resource");
                Err(failure)
            }
            ProbeFailure::Denied { errno } => {
                if matches!(prior, PriorDir::Denied) {
                    tracing::trace!(?anchor, ?read, ?e, "probe_subtree: still denied");
                } else {
                    tracing::debug!(?anchor, ?read, ?e, "probe_subtree: the level is denied");
                }
                Ok(Reached::Denied { errno })
            }
        }
    }
}

/// One `lstat` of a file anchor. Anything but a regular file is `Vanished`, as is nothing at all.
pub(super) fn probe_anchor_file(target_path: &Path) -> ProbeOutcome {
    let meta = match std::fs::symlink_metadata(target_path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return ProbeOutcome::Vanished,
        Err(e) => return ProbeOutcome::Failed(ProbeFailure::from_io(&e)),
    };
    if !meta.is_file() {
        return ProbeOutcome::Vanished;
    }
    // `is_file` upholds `from_metadata`'s non-directory precondition.
    ProbeOutcome::AnchorOk(LeafEntry::from_metadata(&meta))
}

/// The walk of the directory at `target_path` under `config`, measured from `anchor_path`, over
/// `baseline`, the Profile's earlier listing of it.
pub(super) fn probe_subtree(
    target_path: &Path,
    anchor_path: &Path,
    config: &ScanConfig,
    captured_with: u64,
    baseline: Option<&Arc<DirSnapshot>>,
    obligation: &ProofObligation,
    forced: bool,
) -> ProbeOutcome {
    let root_meta = match std::fs::symlink_metadata(target_path) {
        Ok(meta) if meta.is_dir() => DirMeta::from_metadata(&meta),
        Ok(_) => return ProbeOutcome::Vanished,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return ProbeOutcome::Vanished,
        Err(e) => return ProbeOutcome::Failed(ProbeFailure::from_io(&e)),
    };
    let ctx = WalkContext {
        anchor_path,
        config,
        captured_with,
        obligation,
        forced,
        root_dev: root_meta.fs_id().device(),
    };
    let prior = PriorDir::root(baseline, root_meta.fs_id());
    let mut denied = BTreeSet::new();
    match snapshot_dir(&ctx, target_path, root_meta, prior, &mut denied) {
        Ok(Reached::Listed(snapshot)) => ProbeOutcome::SubtreeProven {
            snapshot,
            authority: ProofAuthority::certify(denied),
        },
        Ok(Reached::Gone) => ProbeOutcome::Vanished,
        Ok(Reached::Denied { errno }) => ProbeOutcome::Failed(ProbeFailure::Denied { errno }),
        Err(failure) => ProbeOutcome::Failed(failure),
    }
}

/// A descent's structural query: does `segment` exist under `prefix`, and as what kind? On a miss, a
/// second `lstat`, of the prefix, tells what is missing — the segment alone (`kind: None`), or the
/// prefix itself (`Vanished`) — and an unreadable prefix is `Failed`, a non-observation rather than
/// an absence. The two `lstat`s are not atomic; the next event re-probes whatever the second saw.
pub(super) fn probe_descent(prefix: &Path, segment: &str) -> ProbeOutcome {
    match std::fs::symlink_metadata(prefix.join(segment)) {
        Ok(meta) => ProbeOutcome::SegmentObserved {
            kind: Some(EntryKind::from(meta.file_type())),
        },
        // `ENOTDIR`: a component on the way stopped being a directory.
        Err(e)
            if matches!(
                e.kind(),
                io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
            ) =>
        {
            match std::fs::symlink_metadata(prefix) {
                Ok(meta) if meta.is_dir() => ProbeOutcome::SegmentObserved { kind: None },
                Ok(_) => ProbeOutcome::Vanished,
                Err(e) if e.kind() == io::ErrorKind::NotFound => ProbeOutcome::Vanished,
                Err(e) => ProbeOutcome::Failed(ProbeFailure::from_io(&e)),
            }
        }
        Err(e) => ProbeOutcome::Failed(ProbeFailure::from_io(&e)),
    }
}

/// The listing of the directory at `path` — `root_meta` its `lstat`, `prior` what the earlier
/// snapshot holds for it — naming in `denied` each directory the walk had cause to read and could
/// not.
fn snapshot_dir(
    ctx: &WalkContext<'_>,
    path: &Path,
    root_meta: DirMeta,
    prior: PriorDir<'_>,
    denied: &mut BTreeSet<Arc<Path>>,
) -> Result<Reached<Arc<DirSnapshot>>, ProbeFailure> {
    if let Some(listing) = ctx.mtime_skip(path, root_meta, prior) {
        return Ok(Reached::Listed(Arc::clone(listing)));
    }
    Ok(match read_level(ctx, path, prior)? {
        // The level is known whole before any child is entered, so whatever a child answers can no
        // longer deny the level, and everything `denied` names is an entry of the snapshot.
        Reached::Listed(Level {
            mut entries,
            subdirs,
        }) => {
            for sub in subdirs {
                let fs_id = sub.meta.fs_id();
                let reached = snapshot_dir(
                    ctx,
                    &sub.path,
                    sub.meta,
                    prior.child(&sub.name, fs_id),
                    denied,
                )?;
                if let Some(child) = reached.into_dir_child(fs_id) {
                    entries.insert(sub.name, ChildEntry::Dir(child));
                }
            }
            Reached::Listed(Arc::new(DirSnapshot::new(
                root_meta,
                ctx.captured_with,
                entries,
            )))
        }
        Reached::Gone => Reached::Gone,
        Reached::Denied { errno } => ctx.stand_in(path, root_meta, prior, errno, denied),
    })
}

/// Read the level of the directory at `path` whole, through the shape's gates; a leaf inherits its
/// hash from `prior`'s listing where its identity matches. It takes no denial set, so it can enter
/// no other directory. The directory is closed when it returns.
fn read_level(
    ctx: &WalkContext<'_>,
    path: &Path,
    prior: PriorDir<'_>,
) -> Result<Reached<Level>, ProbeFailure> {
    let dirents = match std::fs::read_dir(path) {
        Ok(dirents) => dirents,
        Err(e) => return ctx.level_fault(path, &e, prior),
    };
    let mut level = Level::default();
    for dirent in dirents {
        let child_path = match dirent {
            Ok(dirent) => dirent.path(),
            Err(e) => return ctx.level_fault(path, &e, prior),
        };
        // No snapshot key, chain, pattern or attach path is anything but UTF-8, so nothing can name
        // this entry or what lies beneath it: it is out of scope, and the level stays whole.
        let Some(name) = child_path.file_name().and_then(OsStr::to_str) else {
            tracing::debug!(
                ?child_path,
                "probe_subtree: a name that is not UTF-8 is out of scope"
            );
            continue;
        };
        let Ok(rel) = child_path.strip_prefix(ctx.anchor_path) else {
            return Err(walker_fault(&child_path, "a dirent outside the anchor"));
        };
        let depth = u32::try_from(rel.components().count()).unwrap_or(u32::MAX);
        if !ctx.config.accepts_structural(rel, depth) {
            // The engine chains only a path `covers` admits, by this gate from this basis.
            if ctx.chain_through(&child_path) {
                return Err(walker_fault(
                    &child_path,
                    "scope regression: the structural gate dropped a dirent on an obligation chain",
                ));
            }
            continue;
        }
        let meta = match std::fs::symlink_metadata(&child_path) {
            Ok(meta) => meta,
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                tracing::trace!(?child_path, "probe_subtree: gone before its lstat");
                continue;
            }
            // Any other error speaks of the level: `lstat` does not follow the child, so `ENOTDIR`
            // and `ELOOP` say the level is no longer a directory, and the rest leave it unread whole.
            Err(e) => return ctx.level_fault(&child_path, &e, prior),
        };
        // A chained name's kind can flip under an atomic replace: the entry left scope, and its
        // absence differs from the prior that held it.
        if !ctx.config.accepts_kinded(rel, meta.is_dir(), depth) {
            tracing::trace!(?child_path, "probe_subtree: its kind is out of scope");
            continue;
        }
        let name = CompactString::new(name);
        if !meta.is_dir() {
            let inherit = prior
                .listing()
                .and_then(|listing| listing.lookup_leaf(&name));
            let leaf = LeafEntry::from_metadata_or_inherit(&meta, inherit);
            level.entries.insert(name, ChildEntry::Leaf(leaf));
        } else if ctx.should_recurse(depth, meta.dev()) {
            level.subdirs.push(Subdir {
                name,
                path: child_path,
                meta: DirMeta::from_metadata(&meta),
            });
        } else {
            let boundary = DirChild::Uncovered(FsIdentity::from_metadata(&meta));
            level.entries.insert(name, ChildEntry::Dir(boundary));
        }
    }
    Ok(Reached::Listed(level))
}

/// A walker fault — the walk finds its own scope broken, and no filesystem refused anything. Loud
/// in a debug build; in release the walk stops with the answer a worker panic gives.
fn walker_fault(child_path: &Path, fault: &str) -> ProbeFailure {
    debug_assert!(false, "walker fault: {fault}: {}", child_path.display());
    tracing::warn!(
        ?child_path,
        fault,
        "probe_subtree: a walker fault; the walk stops"
    );
    PROBER_FAULT
}

#[cfg(test)]
mod tests {
    use super::{Reached, WalkContext, read_level};
    use specter_core::{PatternSpec, PriorDir, ProofObligation, ScanConfig};
    use std::path::Path;
    use std::sync::Arc;

    /// The first phase classifies a failed open itself, since a race between the `lstat` that named
    /// a directory and its open is the only way a walk meets one: nothing at the path, or no longer
    /// a directory, is gone; a directory the walk may not open is denied.
    #[test]
    fn a_failed_open_is_gone_or_denied() {
        use std::num::NonZeroU32;
        use std::os::unix::fs::PermissionsExt;
        let tmp = tempfile::TempDir::new().unwrap();
        let file = tmp.path().join("file");
        std::fs::write(&file, b"x").unwrap();
        let locked = tmp.path().join("locked");
        std::fs::create_dir(&locked).unwrap();
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o000)).unwrap();

        let config = ScanConfig::builder().max_depth(NonZeroU32::new(1)).build();
        let obligation = ProofObligation::WholeSubtree;
        let ctx = WalkContext {
            anchor_path: tmp.path(),
            config: &config,
            captured_with: 0,
            obligation: &obligation,
            forced: false,
            root_dev: 0,
        };
        let read = |path: &Path| read_level(&ctx, path, PriorDir::Unseen);
        let missing = read(&tmp.path().join("missing"));
        let not_a_dir = read(&file);
        let denied = read(&locked);
        std::fs::set_permissions(&locked, std::fs::Permissions::from_mode(0o755)).unwrap();

        assert!(matches!(missing, Ok(Reached::Gone)));
        assert!(matches!(not_a_dir, Ok(Reached::Gone)));
        assert!(matches!(
            denied,
            Ok(Reached::Denied {
                errno: libc::EACCES
            })
        ));
    }

    /// Mounts cannot be built in a unit test, so the cross-device pin is the predicate composition:
    /// a device-mismatched child still descends under `MatchChain` (a chain's levels may cross
    /// mounts) and refuses under `Subtree`, and the same-device control proves the `dev ==
    /// root_dev` translation is not inverted.
    #[test]
    fn should_recurse_device_gate_is_shape_policy() {
        let obligation = ProofObligation::WholeSubtree;
        let chain_cfg = ScanConfig::MatchChain(Arc::new(
            PatternSpec::parse("/srv/*/data/*/log").expect("test pattern parses"),
        ));
        let chain = WalkContext {
            anchor_path: Path::new("/srv"),
            config: &chain_cfg,
            captured_with: 0,
            obligation: &obligation,
            forced: false,
            root_dev: 1,
        };
        assert!(
            chain.should_recurse(1, 2),
            "MatchChain is device-blind above the terminus",
        );
        assert!(
            !chain.should_recurse(4, 2),
            "the terminus depth never descends",
        );

        let subtree_cfg = ScanConfig::builder().build();
        let subtree = WalkContext {
            config: &subtree_cfg,
            ..chain
        };
        assert!(
            !subtree.should_recurse(1, 2),
            "an unbounded recursive walk refuses a cross-device child",
        );
        assert!(
            subtree.should_recurse(1, 1),
            "same-device control: the device translation is not inverted",
        );
    }
}
