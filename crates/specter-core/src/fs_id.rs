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
//!   `fs_id.device()` against the anchor's `root_dev` to decide recursion.
//! - **Rename targets:** `pair_renames` indexes Created entries by
//!   `fs_id` and pairs them with same-`fs_id` Deletes across the diff.
//!
//! ## Atomicity invariant
//!
//! `inode` and `device` are meaningful only as a pair read from a
//! *single* `stat`: across two `stat`s the kernel may recycle an inode
//! number under the same device, so a torn `(inode, device)` assembled
//! from independent observations names nothing. The fields are private
//! and the sole production constructor, [`FsIdentity::from_metadata`],
//! reads *both* halves from one `&Metadata`. No API assembles an
//! identity from two independent observations, so every consumer of an
//! `FsIdentity` knows the two halves agree *by construction* — the
//! invariant is discharged by the type, not by caller convention.
//!
//! The test-only [`FsIdentity::synthetic`] constructor is the single,
//! explicitly named exception: fixtures need identities the kernel would
//! never co-locate (commonly `device: 0`). It is compiled out of
//! release builds (`cfg(any(test, feature = "testkit"))`).
//!
//! ## Identity tiers
//!
//! This is the *kernel-side* identity layer. The engine-side slot
//! identity is [`crate::ResourceId`] (slotmap-generational, survives
//! delete-and-recreate); the two layers compose but are independent.

use crate::hash::StableHasher;
use std::hash::Hasher;

/// Kernel-side observable identity of an inode at `stat` time.
///
/// Construct via [`FsIdentity::from_metadata`] (production — both halves
/// from one `&Metadata`) or `FsIdentity::synthetic` (test-only). See
/// the module-level docs for the semantics and atomicity invariant.
///
/// ## Digest encoding
///
/// Snapshot digests fold `FsIdentity` **exclusively** through
/// `encode_into` — `inode` then `device`, each as a little-endian
/// `u64`, in declaration order, with no discriminator or length prefix.
/// That order is byte-identical to the historical `inode` then `device`
/// `u64` fold; the `encode_into_matches_inode_then_device` regression
/// pins the equivalence and the snapshot goldens (`compute_leaf_hash` /
/// `compute_dir_hash`) pin it end-to-end, so a future field reorder
/// fires at `cargo nextest` time. `FsIdentity` deliberately has **no**
/// `Hash` impl: the only route to a digest is `encode_into`, mirroring
/// the [`StableHasher`] "own the bytes" discipline — a blanket-`Hash`
/// digest path is unconstructable.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct FsIdentity {
    inode: u64,
    device: u64,
}

impl FsIdentity {
    /// Construct from a single freshly-`stat`ed `Metadata`.
    ///
    /// Both halves are read here, inside the constructor, from the
    /// *same* `&Metadata` — this is what discharges the atomicity
    /// invariant at the type boundary. A value constructor taking
    /// `(inode, device)` would be no stronger than public fields: the
    /// caller could still pass halves sourced from two independent
    /// `stat`s.
    ///
    /// `MetadataExt::ino`/`dev` read fields already populated by the
    /// `stat` the sensor performed; they are *not* syscalls, so this is
    /// consistent with `core`'s no-I/O discipline (I1).
    #[cfg(unix)]
    #[must_use]
    pub fn from_metadata(meta: &impl std::os::unix::fs::MetadataExt) -> Self {
        Self {
            inode: meta.ino(),
            device: meta.dev(),
        }
    }

    /// The kernel inode number observed at `stat` time.
    #[must_use]
    pub const fn inode(self) -> u64 {
        self.inode
    }

    /// The kernel device number observed at `stat` time.
    #[must_use]
    pub const fn device(self) -> u64 {
        self.device
    }

    /// Test-only constructor from explicit halves, bypassing the
    /// single-`stat` provenance [`FsIdentity::from_metadata`] enforces.
    ///
    /// Fixtures synthesise identities the kernel would never co-locate
    /// (commonly `device: 0`); the seal keeps that affordance out of the
    /// production surface. Compiled only under `cfg(test)` or the
    /// `testkit` feature.
    #[cfg(any(test, feature = "testkit"))]
    #[must_use]
    pub const fn synthetic(inode: u64, device: u64) -> Self {
        Self { inode, device }
    }
}

/// Fold a [`FsIdentity`] into a stable digest: `inode` then `device`,
/// each as a little-endian `u64`.
///
/// The single named route for committing `FsIdentity` to a digest —
/// the seam analogue beside the type (the kernel identity is the
/// domain knowledge the primitive-only [`StableHasher`] deliberately
/// does not carry). Byte-identical to the historical `inode` then
/// `device` `u64` fold on every target.
pub fn encode_into<H: Hasher>(id: FsIdentity, h: &mut StableHasher<H>) {
    h.put_u64(id.inode);
    h.put_u64(id.device);
}

#[cfg(test)]
mod tests {
    use super::{FsIdentity, encode_into};

    /// `Ord` is inode-first by declaration order. Not externally
    /// load-bearing (the `pair_renames` BTreeMap is never iterated),
    /// but pinning the ordering ensures a future field reorder gets
    /// caught here rather than as a subtle behaviour drift.
    #[test]
    fn ord_compares_inode_first_then_device() {
        let a = FsIdentity::synthetic(1, 99);
        let b = FsIdentity::synthetic(2, 0);
        assert!(a < b, "inode dominates ord even when a.device > b.device");

        let c = FsIdentity::synthetic(1, 99);
        let d = FsIdentity::synthetic(1, 100);
        assert!(c < d, "device breaks ties when inodes are equal");
    }

    /// `encode_into` reproduces the historical `inode` then `device`
    /// `u64` byte stream — each as a little-endian `u64`, in declaration
    /// order, with no discriminator or length prefix. Production folds
    /// `FsIdentity` exclusively through this seam encoder; pinning it
    /// against the legacy `inode.hash(h); device.hash(h)` reference
    /// proves every persisted 128-bit snapshot golden survives the
    /// migration off `derive(Hash)`.
    #[test]
    fn encode_into_matches_inode_then_device() {
        use siphasher::sip128::{Hasher128, SipHasher24 as RawSip128};
        use std::hash::Hash;

        let id = FsIdentity::synthetic(0x1234_5678_9abc_def0, 0xfedc_ba98_7654_3210);

        let mut seam = crate::hash::hasher_128();
        encode_into(id, &mut seam);

        let mut reference = RawSip128::new();
        id.inode().hash(&mut reference);
        id.device().hash(&mut reference);

        assert_eq!(
            seam.finish_u128(),
            u128::from(reference.finish128()),
            "encode_into diverged from the historical inode-then-device \
             fold; every snapshot golden would shift",
        );
    }
}
