//! Pure fixture constructors shared across engine, sensor, and actuator test suites.
//!
//! These build `core` values only — no `Engine`, no I/O. They are the single canonical shape for
//! fixtures across the test suites, so a fixture's layout cannot silently drift between files.

use crate::testkit::single_exec_program;
use crate::{
    ActionProgram, ArgPart, ArgTemplate, ChildEntry, DirChild, DirMeta, DirSnapshot,
    DirtyProvenance, EntryKind, FsIdentity, LeafEntry, ProbeOutcome, ProfileId, ProofAuthority,
    ResourceId,
};
use compact_str::CompactString;
use slotmap::SlotMap;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

/// Build a flat single-level directory snapshot from `(name, kind, inode)` triples.
///
/// Every leaf gets `size = 0`, `mtime = UNIX_EPOCH`, `device = 0`; a `Dir` child is stored
/// `Uncovered` (the walker did not recurse). The root meta is the zero sentinel. This is the
/// synthetic shape engine tests use to drive verdicts by hash equality — two `dir_snap` calls with
/// equal children hash equal.
///
/// Names must be single path components: a `'/'` is a fixture bug (a nested key never round-trips
/// through the `BTreeMap<CompactString, _>`), caught loudly in dev/CI and inert in release — the
/// same discipline as the engine's own tripwires.
#[must_use]
pub fn dir_snap(children: &[(&str, EntryKind, u64)]) -> Arc<DirSnapshot> {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for &(name, kind, inode) in children {
        debug_assert!(
            !name.contains('/'),
            "dir_snap: '{name}' must be a single path component, not a nested path",
        );
        let child = match kind {
            EntryKind::Dir => ChildEntry::Dir(DirChild::Uncovered(FsIdentity::synthetic(inode, 0))),
            _ => ChildEntry::Leaf(LeafEntry::synthetic(
                kind,
                0,
                UNIX_EPOCH,
                FsIdentity::synthetic(inode, 0),
            )),
        };
        map.insert(CompactString::new(name), child);
    }
    Arc::new(DirSnapshot::new(
        DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(0, 0)),
        0,
        map,
    ))
}

/// A synthetic file `LeafEntry` for [`anchor_ok`] / File-anchored quiescence proofs.
///
/// The leaf analogue of [`dir_snap`]'s per-child construction (`size = 0`, `mtime = UNIX_EPOCH`,
/// `device = 0`; only `kind` and `inode` identify it). Two equal-arg calls hash equal, so paired
/// samples through the verdict floor's hash channel agree by construction.
#[must_use]
pub fn file_leaf(kind: EntryKind, inode: u64) -> LeafEntry {
    LeafEntry::synthetic(kind, 0, UNIX_EPOCH, FsIdentity::synthetic(inode, 0))
}

/// Build a [`DirtyProvenance`] from `(slot, absolute-path)` pairs — the canonical fixture for the
/// Standard pre-fire obligation / scope projection (`chains`, `lca_path`) and `pre_fire_target`.
///
/// Mirrors the production ingest contract exactly: each pair is one [`DirtyProvenance::note`] in
/// slice order, so a repeated `ResourceId` is last-writer-wins just as a repeat `FsEvent` for one
/// slot would be. Paths must be **absolute** — production captures a root-materialised `Arc<Path>`,
/// and the component-LCA relies on every value sharing at least the root; a relative path is a
/// fixture bug, caught loudly in dev/CI and inert in release (the same discipline as [`dir_snap`]'s
/// single-component-name check).
#[must_use]
pub fn dirty_provenance(entries: &[(ResourceId, &str)]) -> DirtyProvenance {
    let mut dirty = DirtyProvenance::new();
    for &(id, path) in entries {
        debug_assert!(
            path.starts_with('/'),
            "dirty_provenance: '{path}' must be an absolute path \
             (production captures a root-materialised Arc<Path>)",
        );
        dirty.note(id, Arc::from(Path::new(path)));
    }
    dirty
}

/// A `Subtree` outcome whose walk discharged its obligation.
///
/// The overwhelmingly common engine-test shape (a fully-read, settled subtree): shorthand for the
/// `SubtreeProven { snapshot, authority: ProofAuthority::Authoritative }` literal.
#[must_use]
pub const fn proven(snapshot: Arc<DirSnapshot>) -> ProbeOutcome {
    ProbeOutcome::SubtreeProven {
        snapshot,
        authority: ProofAuthority::Authoritative,
    }
}

/// A `Descent` enumeration outcome — one prefix level, no proof obligation (structural query, not a
/// quiescence observation).
#[must_use]
pub const fn enumerated(snapshot: Arc<DirSnapshot>) -> ProbeOutcome {
    ProbeOutcome::DirEnumerated(snapshot)
}

/// An `AnchorOk` outcome — a File/Symlink anchor's `lstat` result.
///
/// Completes the success-outcome trio with [`proven`] (`SubtreeProven`) and [`enumerated`]
/// (`DirEnumerated`); pair with [`file_leaf`] for a File-anchored quiescence sample (hash-channel
/// pair when the carrier is engaged).
#[must_use]
pub const fn anchor_ok(leaf: LeafEntry) -> ProbeOutcome {
    ProbeOutcome::AnchorOk(leaf)
}

/// The canonical no-op action program (`/bin/true`, single exec).
///
/// The shape every engine/actuator test that does not assert on argv wants; operationally identical
/// to config-lowering one `exec = ["/bin/true"]`.
#[must_use]
pub fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

/// Mint one fresh `ProfileId` from a throwaway slotmap — for sensor / prober tests that need a
/// correlation owner without a live `Engine`.
#[must_use]
pub fn fresh_profile_id() -> ProfileId {
    let mut sm = SlotMap::<ProfileId, ()>::with_key();
    sm.insert(())
}

/// Mint `n` distinct fresh `ProfileId`s from one throwaway slotmap.
#[must_use]
pub fn fresh_profile_ids(n: usize) -> Vec<ProfileId> {
    let mut sm = SlotMap::<ProfileId, ()>::with_key();
    (0..n).map(|_| sm.insert(())).collect()
}
