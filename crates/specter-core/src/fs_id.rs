//! Kernel-side filesystem identity — the `(inode, device)` pair the
//! workspace uses to detect delete-and-recreate-at-same-path, cross-
//! filesystem boundaries, and rename targets across snapshots.
//!
//! ## Semantics
//!
//! POSIX guarantees: `(inode, device)` uniquely identifies an inode
//! *within one filesystem at one point in time*. It is **not** a stable
//! cross-time identifier — after `unlink`, the kernel may reuse the
//! inode number on a subsequent `creat` under the same `device`. Snapshot
//! atoms carry this pair to detect:
//!
//! - **Delete-and-recreate-at-same-path:** prior and fresh have the same
//!   segment but different `fs_id`, so `diff_same_name` emits
//!   Deleted + Created (not Modified).
//! - **Cross-filesystem boundary:** the walker compares each subdir's
//!   `fs_id.device` against the anchor's `root_dev` to decide recursion.
//! - **Rename targets:** `pair_renames` indexes Created entries by
//!   `fs_id` and pairs them with same-`fs_id` Deletes across the diff.
//!
//! ## Atomicity invariant
//!
//! Both fields originate from a *single* `lstat` (or `MetadataExt`)
//! call — they cannot be sourced from independent syscalls. Wrapping
//! the pair in this struct encodes that invariant in the type system:
//! a function consuming `FsIdentity` knows the two halves agree.
//!
//! ## Identity tiers
//!
//! This is the *kernel-side* identity layer. The engine-side slot
//! identity is [`crate::ResourceId`] (slotmap-generational, survives
//! delete-and-recreate); the two layers compose but are independent.

use std::hash::Hash;

/// Kernel-side observable identity of an inode at `lstat` time.
///
/// See the module-level docs for the semantics and atomicity invariant.
///
/// ## Hash byte equivalence
///
/// `#[derive(Hash)]` calls `Hash::hash` on each field in declaration
/// order with no discriminator byte and no length prefix. The byte
/// stream emitted by `fs_id.hash(h)` is therefore byte-identical to
/// the historical `inode.hash(h); device.hash(h);` sequence — pinned
/// by the `fs_identity_hash_matches_inode_then_device` regression
/// test below so a future Rust derive change or field reorder fires
/// at `cargo nextest` time. The 128-bit snapshot hashes
/// (`compute_leaf_hash`, `compute_dir_hash`) depend on this equivalence
/// to preserve goldens across the migration.
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct FsIdentity {
    pub inode: u64,
    pub device: u64,
}

#[cfg(test)]
mod tests {
    use super::FsIdentity;
    use crate::hash::{Hasher128Ext, hasher_128};
    use std::hash::Hash;

    /// Load-bearing regression: `#[derive(Hash)]` on `FsIdentity` must
    /// produce the same byte stream as `inode.hash(h); device.hash(h);`
    /// in declaration order. The snapshot hash collapse depends on this
    /// equivalence to preserve every existing 128-bit golden.
    ///
    /// A failure here means either Rust's derive(Hash) changed semantics
    /// (struct discriminator added, length prefix added, …) or the
    /// declaration order of `FsIdentity`'s fields drifted from
    /// `(inode, device)`. Both would silently invalidate every persisted
    /// snapshot hash; this assertion fires at nextest time so the breakage
    /// is caught before a release.
    #[test]
    fn fs_identity_hash_matches_inode_then_device() {
        let id = FsIdentity {
            inode: 0x1234_5678_9abc_def0,
            device: 0xfedc_ba98_7654_3210,
        };
        let mut combined = hasher_128();
        id.hash(&mut combined);
        let mut sequential = hasher_128();
        id.inode.hash(&mut sequential);
        id.device.hash(&mut sequential);
        assert_eq!(
            combined.finish_128_u128(),
            sequential.finish_128_u128(),
            "FsIdentity derive(Hash) diverged from inode-then-device fold; \
             snapshot goldens will break",
        );
    }

    #[test]
    fn default_is_zero_pair() {
        let id = FsIdentity::default();
        assert_eq!(id.inode, 0);
        assert_eq!(id.device, 0);
    }

    /// `Ord` is inode-first by declaration order. Not externally
    /// load-bearing (the `pair_renames` BTreeMap is never iterated),
    /// but pinning the ordering ensures a future field reorder gets
    /// caught here rather than as a subtle behaviour drift.
    #[test]
    fn ord_compares_inode_first_then_device() {
        let a = FsIdentity {
            inode: 1,
            device: 99,
        };
        let b = FsIdentity {
            inode: 2,
            device: 0,
        };
        assert!(a < b, "inode dominates ord even when a.device > b.device");

        let c = FsIdentity {
            inode: 1,
            device: 99,
        };
        let d = FsIdentity {
            inode: 1,
            device: 100,
        };
        assert!(c < d, "device breaks ties when inodes are equal");
    }
}
