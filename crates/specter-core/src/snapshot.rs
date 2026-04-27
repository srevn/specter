//! Snapshot types — hierarchical (`tree.rs`).
//!
//! [`EntryKind`] (this file) is the leaf-kind enum that flows through the
//! Diff API ([`crate::diff::EntryRef::kind`]) and through `LeafEntry.kind`.
//! The hierarchical `DirSnapshot` / `LeafEntry` / `Snapshot` types live in
//! [`tree`].
//!
//! `#[repr(u8)]` with explicit discriminants pins the on-the-wire encoding
//! for `leaf_hash` / `dir_hash` — reordering is a breaking change visible
//! in the golden tests.

pub mod tree;

/// Filesystem entry kind. `#[repr(u8)]` pins the hash encoding (folded as
/// `kind as u8` in `leaf_hash`).
#[repr(u8)]
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum EntryKind {
    File = 0,
    Dir = 1,
    Symlink = 2,
    Other = 3,
}
