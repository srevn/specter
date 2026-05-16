//! `Diff`, `EntryRef`, `Rename` — the flat delta type the Effect API
//! consumes.
//!
//! The classification function lives in [`crate::snapshot::tree::diff_tree`]
//! — it walks two parallel `Snapshot` trees lock-step and emits this flat
//! shape.
//!
//! Output ordering: each list is in stable depth-first pre-order —
//! `BTreeMap` iteration (lexicographic within a directory) plus fixed
//! recursion, so a directory entry is immediately followed by its whole
//! subtree, before the directory's lexical siblings. Deterministic and
//! replay-stable (`Diff: PartialEq`) but **not** a flat lexicographic
//! sort of `parent/child` paths. `renamed` follows the baseline-side
//! (deleted) traversal order, not a sort by `from`. `modified` carries
//! the *current* entry's `EntryRef` payload (downstream consumers read
//! the new state).
//!
//! Hardlinks (same inode at multiple segments) collide in the
//! `fs_id` rename map; later entry wins. Documented v1 limitation; the
//! `hardlink_no_panic` property test confirms graceful degradation.

use crate::fs_id::FsIdentity;
use crate::snapshot::EntryKind;
use compact_str::CompactString;
use smallvec::SmallVec;

/// Classification of entry-level differences between two snapshots.
///
/// Each list is in stable depth-first pre-order (not a flat-lex sort);
/// see module docs.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct Diff {
    pub created: SmallVec<[EntryRef; 4]>,
    pub deleted: SmallVec<[EntryRef; 4]>,
    pub modified: SmallVec<[EntryRef; 4]>,
    pub renamed: SmallVec<[Rename; 4]>,
}

impl Diff {
    /// `true` iff every list is empty — the snapshots agree on every entry.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.created.is_empty()
            && self.deleted.is_empty()
            && self.modified.is_empty()
            && self.renamed.is_empty()
    }
}

/// Lightweight reference into a snapshot entry — the shape downstream
/// consumers need (`segment` for path joining, `kind` for File/Dir
/// dispatch, `fs_id` for cross-snapshot identity).
///
/// The `fs_id` field carries both inode and device, so the diff atom is
/// faithful to the kernel's identity model across multi-mount setups
/// (the snapshot walker already keys cross-filesystem boundary detection
/// on the device half). The actuator's `SPECTER_DIFF_PATH` wire format
/// currently exposes only the inode half (one identifier per entry,
/// matching `stat -c %i`); device exposure waits on a paired
/// `$SPECTER_DEVICE` resolver decision.
#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct EntryRef {
    pub segment: CompactString,
    pub kind: EntryKind,
    pub fs_id: FsIdentity,
}

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct Rename {
    pub from: EntryRef,
    pub to: EntryRef,
}
