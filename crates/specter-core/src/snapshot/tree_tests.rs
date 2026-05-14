use super::{
    ChildEntry, DirChild, DirMeta, DirSnapshot, LeafEntry, SpliceResult, TreeSnapshot, diff_tree,
    splice,
};
use crate::diff::{Diff, EntryRef, Rename};
use crate::fs_id::FsIdentity;
use crate::ids::ResourceId;
use crate::output::StepOutput;
use crate::resource::ResourceRole;
use crate::snapshot::EntryKind;
use crate::tree::Tree;
use compact_str::CompactString;
use proptest::prelude::*;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, UNIX_EPOCH};

// ---------------------------------------------------------------------------
// Fixture helpers
// ---------------------------------------------------------------------------

fn meta(mtime_secs: u64, inode: u64, device: u64) -> DirMeta {
    DirMeta {
        mtime: UNIX_EPOCH + Duration::from_secs(mtime_secs),
        fs_id: FsIdentity { inode, device },
    }
}

fn leaf(kind: EntryKind, size: u64, mtime_secs: u64, inode: u64, device: u64) -> LeafEntry {
    LeafEntry::new(
        kind,
        size,
        UNIX_EPOCH + Duration::from_secs(mtime_secs),
        FsIdentity { inode, device },
    )
}

fn dir(inode: u64, device: u64, subtree: Option<Arc<DirSnapshot>>) -> ChildEntry {
    match subtree {
        Some(s) => ChildEntry::Dir(DirChild::Covered(s)),
        None => ChildEntry::Dir(DirChild::Uncovered(FsIdentity { inode, device })),
    }
}

fn make_dir(
    root_meta: DirMeta,
    captured_with: u64,
    entries: BTreeMap<CompactString, ChildEntry>,
) -> Arc<DirSnapshot> {
    Arc::new(DirSnapshot::new(root_meta, captured_with, entries))
}

fn name(s: &str) -> CompactString {
    CompactString::new(s)
}

/// Extract the inner `Arc<DirSnapshot>` from a `ChildEntry`. Panics if
/// the entry is a Leaf or an uncovered Dir — only used in fixtures
/// where the structure is statically known.
fn dir_subtree(c: &ChildEntry) -> &Arc<DirSnapshot> {
    match c {
        ChildEntry::Dir(DirChild::Covered(s)) => s,
        ChildEntry::Dir(DirChild::Uncovered(_)) => panic!("expected covered Dir entry"),
        ChildEntry::Leaf(_) => panic!("expected Dir entry, got Leaf"),
    }
}

/// Build a chain `anchor → a → b → c → ...` in the given Tree, returning
/// the leaf id along with each level's id. Each segment becomes a `User`
/// role; this matches what `Tree::ensure_path` does for the leaf, and is
/// fine for tests that don't rely on the role distinction.
fn ensure_chain(tree: &mut Tree, segments: &[&str]) -> Vec<ResourceId> {
    let mut ids = Vec::with_capacity(segments.len());
    let mut cur: Option<ResourceId> = None;
    for s in segments {
        let id = tree.ensure(cur, s, ResourceRole::User);
        ids.push(id);
        cur = Some(id);
    }
    ids
}

// ---------------------------------------------------------------------------
// DirSnapshot construction
// ---------------------------------------------------------------------------

#[test]
fn dir_snapshot_new_empty_well_formed() {
    let m = meta(1, 100, 1);
    let d = make_dir(m, 7, BTreeMap::new());
    assert_eq!(d.root_meta, m);
    assert_eq!(d.captured_with, 7);
    assert!(d.entries.is_empty());
}

#[test]
fn dir_snapshot_clone_preserves_dir_hash() {
    // Eager construction makes `dir_hash` a plain field on the snapshot;
    // auto-derived `Clone` copies the field. This test pins that the
    // derived clone preserves the data-derived hash.
    let d = make_dir(meta(1, 100, 1), 0, BTreeMap::new());
    let h = d.dir_hash();
    let cloned = (*d).clone();
    assert_eq!(cloned.dir_hash(), h);
}

#[test]
fn leaf_entry_clone_preserves_leaf_hash() {
    let original = leaf(EntryKind::File, 10, 1, 42, 0);
    let h = original.leaf_hash();
    let cloned = original.clone();
    assert_eq!(cloned.leaf_hash(), h);
    assert_eq!(original.leaf_hash(), h);
}

// Compile-time assertion: the load-bearing concurrency properties of
// `DirSnapshot` / `LeafEntry` and the `Diff` types. Both snapshot types
// are fully immutable post-construction with no interior mutability, so
// `Send + Sync` follow trivially from each field being `Send + Sync`;
// this assertion guards against a future regression that introduces an
// `Rc`, a `*const`, or a `Cell<...>` field into any of them.
#[allow(dead_code)]
const _SEND_SYNC: fn() = || {
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<DirMeta>();
    assert_sync::<DirMeta>();
    assert_send::<LeafEntry>();
    assert_sync::<LeafEntry>();
    assert_send::<DirChild>();
    assert_sync::<DirChild>();
    assert_send::<ChildEntry>();
    assert_sync::<ChildEntry>();
    assert_send::<DirSnapshot>();
    assert_sync::<DirSnapshot>();
    assert_send::<TreeSnapshot>();
    assert_sync::<TreeSnapshot>();
    assert_send::<EntryRef>();
    assert_sync::<EntryRef>();
    assert_send::<Rename>();
    assert_sync::<Rename>();
    assert_send::<Diff>();
    assert_sync::<Diff>();
};

// ---------------------------------------------------------------------------
// DirSnapshot dir_hash
// ---------------------------------------------------------------------------

#[test]
fn dir_hash_deterministic_same_input() {
    let mut e = BTreeMap::new();
    e.insert(
        name("foo"),
        ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 1, 0)),
    );
    let a = make_dir(meta(1, 100, 1), 0, e.clone());
    let b = make_dir(meta(1, 100, 1), 0, e);
    assert_eq!(a.dir_hash(), b.dir_hash());
}

#[test]
fn dir_hash_idempotent_across_calls() {
    let d = make_dir(meta(1, 100, 1), 0, BTreeMap::new());
    let h1 = d.dir_hash();
    let h2 = d.dir_hash();
    let h3 = d.dir_hash();
    assert_eq!(h1, h2);
    assert_eq!(h2, h3);
}

#[test]
fn dir_hash_invariant_under_root_meta_mtime() {
    // `root_meta.mtime` is intentionally absent from the dir_hash fold:
    // filter-aware identity must be independent of the directory's own
    // lstat-mtime, which the kernel bumps for any dirent-block change
    // (including filtered-out entries the user-configured filter removes
    // from `entries`). Two snapshots whose `(captured_with, inode,
    // device, entries)` agree must hash equal regardless of mtime.
    let a = make_dir(meta(1, 100, 1), 0, BTreeMap::new());
    let b = make_dir(meta(2, 100, 1), 0, BTreeMap::new());
    assert_eq!(a.dir_hash(), b.dir_hash());
}

#[test]
fn dir_hash_distinguishes_root_meta_inode() {
    let a = make_dir(meta(1, 100, 1), 0, BTreeMap::new());
    let b = make_dir(meta(1, 101, 1), 0, BTreeMap::new());
    assert_ne!(a.dir_hash(), b.dir_hash());
}

#[test]
fn dir_hash_distinguishes_root_meta_device() {
    let a = make_dir(meta(1, 100, 1), 0, BTreeMap::new());
    let b = make_dir(meta(1, 100, 2), 0, BTreeMap::new());
    assert_ne!(a.dir_hash(), b.dir_hash());
}

#[test]
fn dir_hash_distinguishes_captured_with() {
    let a = make_dir(meta(1, 100, 1), 0, BTreeMap::new());
    let b = make_dir(meta(1, 100, 1), 1, BTreeMap::new());
    assert_ne!(a.dir_hash(), b.dir_hash());
}

#[test]
fn dir_hash_distinguishes_entry_name() {
    let mut ea = BTreeMap::new();
    ea.insert(
        name("foo"),
        ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 1, 0)),
    );
    let mut eb = BTreeMap::new();
    eb.insert(
        name("bar"),
        ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 1, 0)),
    );
    let a = make_dir(meta(1, 100, 1), 0, ea);
    let b = make_dir(meta(1, 100, 1), 0, eb);
    assert_ne!(a.dir_hash(), b.dir_hash());
}

#[test]
fn dir_hash_distinguishes_entry_count() {
    let mut ea = BTreeMap::new();
    ea.insert(
        name("foo"),
        ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 1, 0)),
    );
    let mut eb = ea.clone();
    eb.insert(
        name("bar"),
        ChildEntry::Leaf(leaf(EntryKind::File, 20, 2, 2, 0)),
    );
    let a = make_dir(meta(1, 100, 1), 0, ea);
    let b = make_dir(meta(1, 100, 1), 0, eb);
    assert_ne!(a.dir_hash(), b.dir_hash());
}

#[test]
fn dir_hash_distinguishes_leaf_vs_dir_at_same_name() {
    let mut ea = BTreeMap::new();
    ea.insert(
        name("x"),
        ChildEntry::Leaf(leaf(EntryKind::File, 0, 1, 5, 0)),
    );
    let mut eb = BTreeMap::new();
    eb.insert(name("x"), dir(5, 0, None));
    let a = make_dir(meta(1, 100, 1), 0, ea);
    let b = make_dir(meta(1, 100, 1), 0, eb);
    assert_ne!(a.dir_hash(), b.dir_hash());
}

#[test]
fn dir_hash_distinguishes_subtree_present_vs_none() {
    let inner = make_dir(meta(2, 200, 1), 0, BTreeMap::new());
    let mut ea = BTreeMap::new();
    ea.insert(name("d"), dir(200, 1, Some(inner)));
    let mut eb = BTreeMap::new();
    eb.insert(name("d"), dir(200, 1, None));
    let a = make_dir(meta(1, 100, 1), 0, ea);
    let b = make_dir(meta(1, 100, 1), 0, eb);
    assert_ne!(a.dir_hash(), b.dir_hash());
}

#[test]
fn dir_hash_distinguishes_subtree_content() {
    let mut left_inner_entries = BTreeMap::new();
    left_inner_entries.insert(
        name("file"),
        ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
    );
    let left_inner = make_dir(meta(2, 200, 1), 0, left_inner_entries);
    let mut right_inner_entries = BTreeMap::new();
    right_inner_entries.insert(
        name("file"),
        ChildEntry::Leaf(leaf(EntryKind::File, 99, 1, 7, 0)), // different size
    );
    let right_inner = make_dir(meta(2, 200, 1), 0, right_inner_entries);
    let mut ea = BTreeMap::new();
    ea.insert(name("d"), dir(200, 1, Some(left_inner)));
    let mut eb = BTreeMap::new();
    eb.insert(name("d"), dir(200, 1, Some(right_inner)));
    let a = make_dir(meta(1, 100, 1), 0, ea);
    let b = make_dir(meta(1, 100, 1), 0, eb);
    assert_ne!(a.dir_hash(), b.dir_hash());
}

/// Golden hash — pins the 128-bit `dir_hash` encoding (header layout,
/// length prefix, leaf vs dir tags, lex-by-name fold). Drift here changes
/// every cached `dir_hash` this binary computes; update only this constant
/// after a deliberate review.
#[test]
fn dir_hash_known_good_golden() {
    let mut entries = BTreeMap::new();
    entries.insert(
        name("foo.c"),
        ChildEntry::Leaf(LeafEntry::new(
            EntryKind::File,
            100,
            UNIX_EPOCH + Duration::from_secs(1),
            FsIdentity {
                inode: 42,
                device: 99,
            },
        )),
    );
    let d = make_dir(
        DirMeta {
            mtime: UNIX_EPOCH + Duration::from_secs(7),
            fs_id: FsIdentity {
                inode: 1,
                device: 99,
            },
        },
        13,
        entries,
    );
    assert_eq!(d.dir_hash(), GOLDEN_DIR_HASH);
}

const GOLDEN_DIR_HASH: u128 = 0x689b_b808_8bed_61eb_18e1_e6e8_675b_e51f;

// ---------------------------------------------------------------------------
// LeafEntry leaf_hash
// ---------------------------------------------------------------------------

#[test]
fn leaf_hash_deterministic() {
    let a = leaf(EntryKind::File, 10, 1, 7, 0);
    let b = leaf(EntryKind::File, 10, 1, 7, 0);
    assert_eq!(a.leaf_hash(), b.leaf_hash());
}

#[test]
fn leaf_hash_idempotent() {
    let l = leaf(EntryKind::File, 10, 1, 7, 0);
    let h1 = l.leaf_hash();
    let h2 = l.leaf_hash();
    let h3 = l.leaf_hash();
    assert_eq!(h1, h2);
    assert_eq!(h2, h3);
}

#[test]
fn leaf_hash_distinguishes_kind() {
    let a = leaf(EntryKind::File, 10, 1, 7, 0);
    let b = leaf(EntryKind::Dir, 10, 1, 7, 0);
    assert_ne!(a.leaf_hash(), b.leaf_hash());
}

#[test]
fn leaf_hash_distinguishes_size() {
    let a = leaf(EntryKind::File, 10, 1, 7, 0);
    let b = leaf(EntryKind::File, 11, 1, 7, 0);
    assert_ne!(a.leaf_hash(), b.leaf_hash());
}

#[test]
fn leaf_hash_distinguishes_mtime() {
    let a = leaf(EntryKind::File, 10, 1, 7, 0);
    let b = leaf(EntryKind::File, 10, 2, 7, 0);
    assert_ne!(a.leaf_hash(), b.leaf_hash());
}

#[test]
fn leaf_hash_distinguishes_inode() {
    let a = leaf(EntryKind::File, 10, 1, 7, 0);
    let b = leaf(EntryKind::File, 10, 1, 8, 0);
    assert_ne!(a.leaf_hash(), b.leaf_hash());
}

#[test]
fn leaf_hash_distinguishes_device() {
    let a = leaf(EntryKind::File, 10, 1, 7, 0);
    let b = leaf(EntryKind::File, 10, 1, 7, 1);
    assert_ne!(a.leaf_hash(), b.leaf_hash());
}

/// Golden hash — pins the 128-bit `leaf_hash` encoding (kind tag, size,
/// mtime, inode, device fold). Drift here changes every cached `leaf_hash`
/// this binary computes; update only this constant after a deliberate
/// review.
#[test]
fn leaf_hash_known_good_golden() {
    let l = LeafEntry::new(
        EntryKind::File,
        100,
        UNIX_EPOCH + Duration::from_secs(1),
        FsIdentity {
            inode: 42,
            device: 99,
        },
    );
    assert_eq!(l.leaf_hash(), GOLDEN_LEAF_HASH);
}

const GOLDEN_LEAF_HASH: u128 = 0x8b04_357b_6b61_4546_6947_f1f3_280d_d31b;

// ---------------------------------------------------------------------------
// LeafEntry::new_or_inherit
//
// Under eager construction, `new_or_inherit` is the single safe public
// surface for the walker's per-leaf cache-transfer optimisation. The
// inherited hash is semantically equivalent to a recomputed hash (both
// are pure functions of the identity fields), so positive-case tests
// pin the no-op observable behaviour and negative-case tests pin that
// identity-mismatched baselines do not leak.
// ---------------------------------------------------------------------------

fn fields() -> (EntryKind, u64, std::time::SystemTime, FsIdentity) {
    (
        EntryKind::File,
        10,
        UNIX_EPOCH + Duration::from_secs(1),
        FsIdentity {
            inode: 7,
            device: 0,
        },
    )
}

#[test]
fn new_or_inherit_no_baseline_equals_new() {
    let (kind, size, mtime, fs_id) = fields();
    let fresh = LeafEntry::new(kind, size, mtime, fs_id);
    let inherited = LeafEntry::new_or_inherit(kind, size, mtime, fs_id, None);
    assert_eq!(fresh.leaf_hash(), inherited.leaf_hash());
}

#[test]
fn new_or_inherit_matches_baseline_hash_on_identity_match() {
    let (kind, size, mtime, fs_id) = fields();
    let baseline = LeafEntry::new(kind, size, mtime, fs_id);
    let inherited = LeafEntry::new_or_inherit(kind, size, mtime, fs_id, Some(&baseline));
    assert_eq!(inherited.leaf_hash(), baseline.leaf_hash());
}

#[test]
fn new_or_inherit_returns_fresh_hash_on_any_field_mismatch() {
    // Construct a baseline whose hash genuinely differs from the fresh
    // fields' canonical hash: each call varies exactly one field. The
    // assertion shape is "fresh result equals `LeafEntry::new(fresh fields)`,
    // not baseline" — the constructor must NOT inherit when identity
    // differs, regardless of which field flipped.
    fn assert_fresh(
        baseline_fields: (EntryKind, u64, std::time::SystemTime, FsIdentity),
        fresh_fields: (EntryKind, u64, std::time::SystemTime, FsIdentity),
        label: &str,
    ) {
        let baseline = LeafEntry::new(
            baseline_fields.0,
            baseline_fields.1,
            baseline_fields.2,
            baseline_fields.3,
        );
        let fresh_canonical = LeafEntry::new(
            fresh_fields.0,
            fresh_fields.1,
            fresh_fields.2,
            fresh_fields.3,
        );
        let inherited = LeafEntry::new_or_inherit(
            fresh_fields.0,
            fresh_fields.1,
            fresh_fields.2,
            fresh_fields.3,
            Some(&baseline),
        );
        assert_eq!(
            inherited.leaf_hash(),
            fresh_canonical.leaf_hash(),
            "{label}: inherited must equal fresh canonical hash",
        );
        assert_ne!(
            inherited.leaf_hash(),
            baseline.leaf_hash(),
            "{label}: inherited must NOT equal baseline hash",
        );
    }

    let base = fields();
    assert_fresh(
        base,
        (EntryKind::Symlink, base.1, base.2, base.3),
        "kind mismatch",
    );
    assert_fresh(
        base,
        (base.0, base.1.wrapping_add(1), base.2, base.3),
        "size mismatch",
    );
    assert_fresh(
        base,
        (base.0, base.1, base.2 + Duration::from_secs(1), base.3),
        "mtime mismatch",
    );
    assert_fresh(
        base,
        (
            base.0,
            base.1,
            base.2,
            FsIdentity {
                inode: base.3.inode + 1,
                device: base.3.device,
            },
        ),
        "inode mismatch",
    );
    assert_fresh(
        base,
        (
            base.0,
            base.1,
            base.2,
            FsIdentity {
                inode: base.3.inode,
                device: base.3.device + 1,
            },
        ),
        "device mismatch",
    );
}

// ---------------------------------------------------------------------------
// DirSnapshot::lookup_leaf
// ---------------------------------------------------------------------------

#[test]
fn lookup_leaf_returns_leaf_when_present() {
    let mut entries = BTreeMap::new();
    let l = leaf(EntryKind::File, 10, 1, 7, 0);
    let h = l.leaf_hash();
    entries.insert(name("a.c"), ChildEntry::Leaf(l));
    let d = make_dir(meta(1, 100, 1), 0, entries);
    let got = d.lookup_leaf("a.c").expect("a.c present");
    assert_eq!(got.leaf_hash(), h);
}

#[test]
fn lookup_leaf_returns_none_when_missing() {
    let d = make_dir(meta(1, 100, 1), 0, BTreeMap::new());
    assert!(d.lookup_leaf("missing").is_none());
}

#[test]
fn lookup_leaf_returns_none_for_dir_entry() {
    let mut entries = BTreeMap::new();
    entries.insert(name("a"), dir(7, 0, None));
    let d = make_dir(meta(1, 100, 1), 0, entries);
    assert!(d.lookup_leaf("a").is_none());
}

// ---------------------------------------------------------------------------
// DirSnapshot::lookup_covered_dir
// ---------------------------------------------------------------------------

#[test]
fn lookup_covered_dir_returns_inner_arc_when_covered() {
    let inner = make_dir(meta(2, 200, 1), 0, BTreeMap::new());
    let mut entries = BTreeMap::new();
    entries.insert(name("d"), dir(200, 1, Some(Arc::clone(&inner))));
    let parent = make_dir(meta(1, 100, 1), 0, entries);
    let got = parent.lookup_covered_dir("d").expect("covered dir present");
    assert!(Arc::ptr_eq(got, &inner));
}

#[test]
fn lookup_covered_dir_returns_none_when_missing() {
    let parent = make_dir(meta(1, 100, 1), 0, BTreeMap::new());
    assert!(parent.lookup_covered_dir("missing").is_none());
}

#[test]
fn lookup_covered_dir_returns_none_for_uncovered_dir() {
    let mut entries = BTreeMap::new();
    entries.insert(name("d"), dir(200, 1, None));
    let parent = make_dir(meta(1, 100, 1), 0, entries);
    assert!(parent.lookup_covered_dir("d").is_none());
}

#[test]
fn lookup_covered_dir_returns_none_for_leaf_entry() {
    let mut entries = BTreeMap::new();
    entries.insert(
        name("d"),
        ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
    );
    let parent = make_dir(meta(1, 100, 1), 0, entries);
    assert!(parent.lookup_covered_dir("d").is_none());
}

// ---------------------------------------------------------------------------
// DirChild::fs_id projection
// ---------------------------------------------------------------------------

#[test]
fn dirchild_fs_id_covered_sources_from_root_meta() {
    let inner = make_dir(meta(2, 200, 1), 0, BTreeMap::new());
    let dc = DirChild::Covered(inner);
    assert_eq!(
        dc.fs_id(),
        FsIdentity {
            inode: 200,
            device: 1,
        },
    );
}

#[test]
fn dirchild_fs_id_uncovered_returns_stored_value() {
    let dc = DirChild::Uncovered(FsIdentity {
        inode: 42,
        device: 99,
    });
    assert_eq!(
        dc.fs_id(),
        FsIdentity {
            inode: 42,
            device: 99,
        },
    );
}

// ---------------------------------------------------------------------------
// compute_dir_hash — per-variant tag discrimination
// ---------------------------------------------------------------------------
//
// The three-tag encoding makes covered/uncovered structurally
// distinguishable in the fold. Verify that the discrimination survives
// a "naïve" attack: two snapshots whose `Covered` and `Uncovered`
// payloads coincidentally hash to the same value at the slot, but
// whose variant tags differ.

#[test]
fn dir_hash_distinguishes_covered_empty_from_uncovered_same_fs_id() {
    // Covered with an empty subtree at fs_id (200, 1) — `dir_hash` of
    // the inner is *not* equal to 0 in general, but specifically it
    // differs from the raw fs_id fold the Uncovered arm emits.
    let inner_empty = make_dir(meta(2, 200, 1), 0, BTreeMap::new());
    let mut ea = BTreeMap::new();
    ea.insert(name("d"), dir(200, 1, Some(inner_empty)));
    let mut eb = BTreeMap::new();
    eb.insert(name("d"), dir(200, 1, None));
    let a = make_dir(meta(1, 100, 1), 0, ea);
    let b = make_dir(meta(1, 100, 1), 0, eb);
    assert_ne!(
        a.dir_hash(),
        b.dir_hash(),
        "Covered(empty) and Uncovered at the same fs_id must hash distinctly",
    );
}

// ---------------------------------------------------------------------------
// TreeSnapshot::stable_against
// ---------------------------------------------------------------------------

#[test]
fn stable_against_self_dir() {
    let d = make_dir(meta(1, 100, 1), 0, BTreeMap::new());
    let s = TreeSnapshot::Dir(d);
    assert!(s.stable_against(&s));
}

#[test]
fn stable_against_self_file() {
    let s = TreeSnapshot::File(leaf(EntryKind::File, 10, 1, 1, 0));
    assert!(s.stable_against(&s));
}

#[test]
fn stable_against_distinct_dir_hashes_false() {
    // Two snapshots with distinct entries hash to distinct values; the
    // stability verdict says they are not observably the same.
    let mut entries_a: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    entries_a.insert(
        name("foo"),
        ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
    );
    let a = TreeSnapshot::Dir(make_dir(meta(1, 100, 1), 0, entries_a));
    let b = TreeSnapshot::Dir(make_dir(meta(1, 100, 1), 0, BTreeMap::new()));
    assert!(!a.stable_against(&b));
}

#[test]
fn stable_against_distinct_leaf_hashes_false() {
    let a = TreeSnapshot::File(leaf(EntryKind::File, 10, 1, 1, 0));
    let b = TreeSnapshot::File(leaf(EntryKind::File, 11, 1, 1, 0));
    assert!(!a.stable_against(&b));
}

#[test]
fn stable_against_kind_mismatch_false() {
    let a = TreeSnapshot::Dir(make_dir(meta(1, 100, 1), 0, BTreeMap::new()));
    let b = TreeSnapshot::File(leaf(EntryKind::File, 0, 0, 0, 0));
    assert!(!a.stable_against(&b));
    assert!(!b.stable_against(&a));
}

// ---------------------------------------------------------------------------
// TreeSnapshot::subtree_at
// ---------------------------------------------------------------------------

/// Build a 4-level snapshot anchor → a → b → c, in lock-step with a Tree.
/// Returns (snapshot, tree, ids). Ids is `[anchor, a, b, c]`.
fn build_4_level_tree() -> (TreeSnapshot, Tree, Vec<ResourceId>) {
    let mut tree = Tree::new();
    let ids = ensure_chain(&mut tree, &["anchor", "a", "b", "c"]);

    // c (leaf dir, no children)
    let c_snap = make_dir(meta(4, 4, 0), 7, BTreeMap::new());
    // b contains c
    let mut b_entries = BTreeMap::new();
    b_entries.insert(name("c"), dir(4, 0, Some(Arc::clone(&c_snap))));
    let b_snap = make_dir(meta(3, 3, 0), 7, b_entries);
    // a contains b
    let mut a_entries = BTreeMap::new();
    a_entries.insert(name("b"), dir(3, 0, Some(Arc::clone(&b_snap))));
    let a_snap = make_dir(meta(2, 2, 0), 7, a_entries);
    // anchor contains a + a sibling leaf for off-path testing
    let mut root_entries = BTreeMap::new();
    root_entries.insert(name("a"), dir(2, 0, Some(Arc::clone(&a_snap))));
    root_entries.insert(
        name("z_leaf"),
        ChildEntry::Leaf(leaf(EntryKind::File, 99, 1, 99, 0)),
    );
    let root = make_dir(meta(1, 1, 0), 7, root_entries);

    (TreeSnapshot::Dir(root), tree, ids)
}

#[test]
fn subtree_at_anchor_returns_root() {
    let (snap, tree, ids) = build_4_level_tree();
    let anchor = ids[0];
    let got = snap
        .subtree_at(anchor, anchor, &tree)
        .expect("anchor resolves");
    if let TreeSnapshot::Dir(root) = &snap {
        assert!(Arc::ptr_eq(&got, root));
    } else {
        panic!("expected Dir snapshot");
    }
}

#[test]
fn subtree_at_one_level_deep() {
    let (snap, tree, ids) = build_4_level_tree();
    let anchor = ids[0];
    let a = ids[1];
    let got = snap.subtree_at(anchor, a, &tree).expect("a resolves");
    assert!(got.entries.contains_key("b"));
}

#[test]
fn subtree_at_three_levels_deep() {
    let (snap, tree, ids) = build_4_level_tree();
    let anchor = ids[0];
    let c = ids[3];
    let got = snap.subtree_at(anchor, c, &tree).expect("c resolves");
    assert!(got.entries.is_empty());
}

#[test]
fn subtree_at_returns_arc_ptr_eq_with_internal_subtree() {
    let (snap, tree, ids) = build_4_level_tree();
    let anchor = ids[0];
    let b = ids[2];
    let got = snap.subtree_at(anchor, b, &tree).expect("b resolves");
    if let TreeSnapshot::Dir(root) = &snap {
        let internal_a = dir_subtree(root.entries.get("a").unwrap());
        let internal_b = dir_subtree(internal_a.entries.get("b").unwrap());
        assert!(Arc::ptr_eq(&got, internal_b));
    } else {
        panic!("expected Dir snapshot");
    }
}

#[test]
fn subtree_at_target_outside_anchor_returns_none() {
    let (snap, mut tree, ids) = build_4_level_tree();
    let anchor = ids[0];
    // Add a sibling root with no relation to the anchor's chain.
    let stranger = tree.ensure(None, "stranger", ResourceRole::User);
    assert!(snap.subtree_at(anchor, stranger, &tree).is_none());
}

#[test]
fn subtree_at_target_path_through_leaf_returns_none() {
    let (snap, mut tree, ids) = build_4_level_tree();
    let anchor = ids[0];
    // Descend into z_leaf (a Leaf entry) — chain anchor → z_leaf — and
    // ask for a child *of* z_leaf, which is impossible in tree terms.
    // Synthesise a tree id under z_leaf to drive the path.
    let z_leaf_id = tree.ensure(Some(anchor), "z_leaf", ResourceRole::User);
    let inside_leaf = tree.ensure(Some(z_leaf_id), "inside", ResourceRole::User);
    assert!(
        snap.subtree_at(anchor, inside_leaf, &tree).is_none(),
        "chain through Leaf must yield None",
    );
}

#[test]
fn subtree_at_target_path_through_uncovered_returns_none() {
    let mut tree = Tree::new();
    let ids = ensure_chain(&mut tree, &["anchor", "a", "b"]);
    let anchor = ids[0];
    let a = ids[1];
    let b = ids[2];

    // TreeSnapshot represents anchor with `a` as `DirChild::Uncovered(_)`
    // — the walker stored "a" but did not recurse. Asking for `b`
    // (under uncovered `a`) must return None.
    let mut root_entries = BTreeMap::new();
    root_entries.insert(name("a"), dir(2, 0, None));
    let root = make_dir(meta(1, 1, 0), 7, root_entries);
    let snap = TreeSnapshot::Dir(root);
    assert!(snap.subtree_at(anchor, a, &tree).is_none());
    assert!(snap.subtree_at(anchor, b, &tree).is_none());
}

#[test]
fn subtree_at_stale_target_returns_none() {
    let mut tree = Tree::new();
    let ids = ensure_chain(&mut tree, &["anchor", "a"]);
    let anchor = ids[0];
    let a = ids[1];
    let stale = a;

    // Build snapshot with anchor → a (no subtree).
    let mut root_entries = BTreeMap::new();
    root_entries.insert(name("a"), dir(2, 0, None));
    let root = make_dir(meta(1, 1, 0), 7, root_entries);
    let snap = TreeSnapshot::Dir(root);

    // Vacate a and try to reap (children=0, profiles=0 ⇒ reaps clean).
    let reaped = tree.try_reap(a, &mut StepOutput::default());
    assert!(reaped, "a is reapable in this fixture");
    // Now stale is a fresh-looking id with no live slot.
    assert!(snap.subtree_at(anchor, stale, &tree).is_none());
}

#[test]
fn subtree_at_file_snapshot_returns_none() {
    let snap = TreeSnapshot::File(leaf(EntryKind::File, 0, 0, 0, 0));
    let mut tree = Tree::new();
    let id = tree.ensure(None, "anything", ResourceRole::User);
    assert!(snap.subtree_at(id, id, &tree).is_none());
}

// ---------------------------------------------------------------------------
// splice
// ---------------------------------------------------------------------------

/// Test helper: unwrap a `SpliceResult::Spliced(_)` to its inner snapshot,
/// asserting the splice did NOT report a CrossedUncovered contract
/// violation. Tests that exercise the failure path use the variant match
/// directly.
fn unwrap_spliced(r: SpliceResult) -> TreeSnapshot {
    match r {
        SpliceResult::Spliced(s) => s,
        SpliceResult::CrossedUncovered => panic!("expected Spliced, got CrossedUncovered"),
    }
}

#[test]
fn splice_no_prior_returns_replacement() {
    let mut tree = Tree::new();
    let id = tree.ensure(None, "anchor", ResourceRole::User);
    let r = make_dir(meta(1, 1, 0), 0, BTreeMap::new());
    let s = unwrap_spliced(splice(None, id, id, Arc::clone(&r), &tree));
    if let TreeSnapshot::Dir(d) = s {
        assert!(Arc::ptr_eq(&d, &r));
    } else {
        panic!();
    }
}

#[test]
fn splice_at_anchor_replaces_root() {
    let mut tree = Tree::new();
    let id = tree.ensure(None, "anchor", ResourceRole::User);
    let prior = make_dir(meta(1, 1, 0), 0, BTreeMap::new());
    let mut new_entries = BTreeMap::new();
    new_entries.insert(
        name("x"),
        ChildEntry::Leaf(leaf(EntryKind::File, 0, 0, 7, 0)),
    );
    let replacement = make_dir(meta(2, 1, 0), 0, new_entries);
    let s = unwrap_spliced(splice(
        Some(Arc::clone(&prior)),
        id,
        id,
        Arc::clone(&replacement),
        &tree,
    ));
    if let TreeSnapshot::Dir(d) = s {
        assert!(Arc::ptr_eq(&d, &replacement));
    } else {
        panic!();
    }
}

#[test]
fn splice_at_anchor_equal_hash_keeps_prior_arc() {
    let mut tree = Tree::new();
    let id = tree.ensure(None, "anchor", ResourceRole::User);
    let prior = make_dir(meta(1, 1, 0), 0, BTreeMap::new());
    // Construct a structurally-identical replacement; dir_hash folds the
    // observable identity so hashes match.
    let replacement = make_dir(meta(1, 1, 0), 0, BTreeMap::new());
    assert_eq!(prior.dir_hash(), replacement.dir_hash());
    let s = unwrap_spliced(splice(Some(Arc::clone(&prior)), id, id, replacement, &tree));
    if let TreeSnapshot::Dir(d) = s {
        assert!(
            Arc::ptr_eq(&d, &prior),
            "G7-trivial: equal hash hands back prior Arc",
        );
    } else {
        panic!();
    }
}

#[test]
fn splice_one_level_deep_off_path_arc_ptr_eq() {
    let mut tree = Tree::new();
    let ids = ensure_chain(&mut tree, &["anchor", "a"]);
    let anchor = ids[0];
    let a = ids[1];

    // Sibling subtree "off_path"; we'll splice at `a` and assert the
    // sibling's Arc inside the rebuilt root is the *same* Arc as before.
    let off_path = make_dir(meta(99, 99, 0), 0, BTreeMap::new());
    let prior_a = make_dir(meta(2, 2, 0), 0, BTreeMap::new());
    let mut root_entries = BTreeMap::new();
    root_entries.insert(name("a"), dir(2, 0, Some(Arc::clone(&prior_a))));
    root_entries.insert(name("off_path"), dir(99, 0, Some(Arc::clone(&off_path))));
    let root = make_dir(meta(1, 1, 0), 0, root_entries);

    // Replacement at `a` carries a child the prior didn't — observably
    // different snapshots, distinct dir_hash.
    let mut replacement_entries = BTreeMap::new();
    replacement_entries.insert(
        name("file"),
        ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 7, 0)),
    );
    let replacement = make_dir(meta(2, 2, 0), 0, replacement_entries);
    assert_ne!(prior_a.dir_hash(), replacement.dir_hash());
    let s = unwrap_spliced(splice(Some(root), anchor, a, replacement, &tree));
    let TreeSnapshot::Dir(new_root) = s else {
        panic!()
    };
    let off_path_after = dir_subtree(new_root.entries.get("off_path").unwrap());
    assert!(
        Arc::ptr_eq(off_path_after, &off_path),
        "off-path sibling Arc preserved",
    );
}

#[test]
fn splice_three_levels_deep_off_path_arc_ptr_eq() {
    let mut tree = Tree::new();
    let ids = ensure_chain(&mut tree, &["anchor", "a", "b", "c"]);
    let anchor = ids[0];
    let a = ids[1];
    let c = ids[3];

    // Build a sibling "top_sib" under anchor, and "mid_sib" under a, to
    // assert spine-rebuild preserves both.
    let _top_sib_id = tree.ensure(Some(anchor), "top_sib", ResourceRole::User);
    let _mid_sib_id = tree.ensure(Some(a), "mid_sib", ResourceRole::User);
    let top_sib = make_dir(meta(91, 91, 0), 0, BTreeMap::new());
    let mid_sib = make_dir(meta(92, 92, 0), 0, BTreeMap::new());

    let prior_c = make_dir(meta(4, 4, 0), 0, BTreeMap::new());
    let mut b_entries = BTreeMap::new();
    b_entries.insert(name("c"), dir(4, 0, Some(Arc::clone(&prior_c))));
    let b_snap = make_dir(meta(3, 3, 0), 0, b_entries);
    let mut a_entries = BTreeMap::new();
    a_entries.insert(name("b"), dir(3, 0, Some(Arc::clone(&b_snap))));
    a_entries.insert(name("mid_sib"), dir(92, 0, Some(Arc::clone(&mid_sib))));
    let a_snap = make_dir(meta(2, 2, 0), 0, a_entries);
    let mut root_entries = BTreeMap::new();
    root_entries.insert(name("a"), dir(2, 0, Some(Arc::clone(&a_snap))));
    root_entries.insert(name("top_sib"), dir(91, 0, Some(Arc::clone(&top_sib))));
    let root = make_dir(meta(1, 1, 0), 0, root_entries);

    let replacement_c = make_dir(meta(40, 4, 0), 0, BTreeMap::new());
    let s = unwrap_spliced(splice(Some(root), anchor, c, replacement_c, &tree));
    let TreeSnapshot::Dir(new_root) = s else {
        panic!()
    };
    let top_sib_after = dir_subtree(new_root.entries.get("top_sib").unwrap());
    assert!(
        Arc::ptr_eq(top_sib_after, &top_sib),
        "top-level sibling Arc preserved",
    );
    let new_a = dir_subtree(new_root.entries.get("a").unwrap());
    let mid_sib_after = dir_subtree(new_a.entries.get("mid_sib").unwrap());
    assert!(
        Arc::ptr_eq(mid_sib_after, &mid_sib),
        "mid-level sibling Arc preserved",
    );
}

#[test]
fn splice_equal_hash_at_leaf_keeps_prior() {
    let mut tree = Tree::new();
    let ids = ensure_chain(&mut tree, &["anchor", "a"]);
    let anchor = ids[0];
    let a = ids[1];
    let prior_a = make_dir(meta(2, 2, 0), 0, BTreeMap::new());
    let mut root_entries = BTreeMap::new();
    root_entries.insert(name("a"), dir(2, 0, Some(Arc::clone(&prior_a))));
    let root = make_dir(meta(1, 1, 0), 0, root_entries);

    // Replacement at `a` is structurally identical (same metadata + entries).
    let replacement_a = make_dir(meta(2, 2, 0), 0, BTreeMap::new());
    assert_eq!(prior_a.dir_hash(), replacement_a.dir_hash());
    let s = unwrap_spliced(splice(
        Some(Arc::clone(&root)),
        anchor,
        a,
        replacement_a,
        &tree,
    ));
    let TreeSnapshot::Dir(new_root) = s else {
        panic!()
    };
    assert!(
        Arc::ptr_eq(&new_root, &root),
        "G7 leaf-equal-hash propagates Arc::ptr_eq up the spine",
    );
}

#[test]
fn splice_equal_hash_at_intermediate_keeps_prior_spine() {
    let mut tree = Tree::new();
    let ids = ensure_chain(&mut tree, &["anchor", "a", "b"]);
    let anchor = ids[0];
    let b = ids[2];
    let b_snap = make_dir(meta(3, 3, 0), 0, BTreeMap::new());
    let mut a_entries = BTreeMap::new();
    a_entries.insert(name("b"), dir(3, 0, Some(Arc::clone(&b_snap))));
    let a_snap = make_dir(meta(2, 2, 0), 0, a_entries);
    let mut root_entries = BTreeMap::new();
    root_entries.insert(name("a"), dir(2, 0, Some(Arc::clone(&a_snap))));
    let root = make_dir(meta(1, 1, 0), 0, root_entries);

    // Replacement at `b` matches prior_b → splice_dir at `b` returns
    // Arc::clone(prior_b); recursion at `a` sees ptr_eq → returns
    // Arc::clone(a); top sees ptr_eq → returns prior root.
    let replacement_b = make_dir(meta(3, 3, 0), 0, BTreeMap::new());
    let s = unwrap_spliced(splice(
        Some(Arc::clone(&root)),
        anchor,
        b,
        replacement_b,
        &tree,
    ));
    let TreeSnapshot::Dir(new_root) = s else {
        panic!()
    };
    assert!(
        Arc::ptr_eq(&new_root, &root),
        "spine kept across two levels"
    );
}

#[test]
fn splice_replacement_changes_dir_hash_uncached_recompute_correct() {
    let mut tree = Tree::new();
    let ids = ensure_chain(&mut tree, &["anchor", "a"]);
    let anchor = ids[0];
    let a = ids[1];
    let prior_a = make_dir(meta(2, 2, 0), 0, BTreeMap::new());
    let mut root_entries = BTreeMap::new();
    root_entries.insert(name("a"), dir(2, 0, Some(Arc::clone(&prior_a))));
    let root = make_dir(meta(1, 1, 0), 0, root_entries);
    let prior_root_hash = root.dir_hash();

    // Replacement at `a` carries a child the prior didn't — observably
    // different, so the spine rebuild produces a different root hash.
    let mut replacement_entries = BTreeMap::new();
    replacement_entries.insert(
        name("file"),
        ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 7, 0)),
    );
    let replacement_a = make_dir(meta(2, 2, 0), 0, replacement_entries);
    let s = unwrap_spliced(splice(Some(root), anchor, a, replacement_a, &tree));
    let TreeSnapshot::Dir(new_root) = s else {
        panic!()
    };
    // The new root must have a fresh dir_hash that differs from prior.
    assert_ne!(new_root.dir_hash(), prior_root_hash);
}

#[test]
fn splice_target_outside_observed_returns_crossed_uncovered() {
    // Target is in the Tree but outside the prior anchor's subtree.
    // Pre-PR behavior: wholesale-replace with `replacement` (corrupted
    // root). New behavior: SpliceResult::CrossedUncovered — the caller
    // keeps its own Arc handle to the prior view (this test cloned it
    // before passing to splice), preserving Profile.current's anchor-
    // rooted invariant. Caller (graft) emits
    // Diagnostic::SpliceCrossedUncovered.
    let mut tree = Tree::new();
    let ids = ensure_chain(&mut tree, &["anchor"]);
    let anchor = ids[0];
    let stranger = tree.ensure(None, "stranger", ResourceRole::User);
    let prior = make_dir(meta(1, 1, 0), 0, BTreeMap::new());
    let replacement = make_dir(meta(2, 2, 0), 0, BTreeMap::new());
    let prior_strong_before = Arc::strong_count(&prior);
    let s = splice(
        Some(Arc::clone(&prior)),
        anchor,
        stranger,
        Arc::clone(&replacement),
        &tree,
    );
    assert!(
        matches!(s, SpliceResult::CrossedUncovered),
        "target outside observed subtree ⇒ CrossedUncovered",
    );
    // splice consumed its Arc on the failure path; the caller's handle
    // (`prior`) survives at its pre-call strong count.
    assert_eq!(
        Arc::strong_count(&prior),
        prior_strong_before,
        "caller's prior Arc handle survives the splice consume",
    );
}

#[test]
fn splice_target_chain_through_uncovered_returns_crossed_uncovered() {
    // Snapshot has anchor → a, but a's subtree is None (uncovered). The
    // splice path reaches the uncovered intermediate and cannot navigate
    // further. New behavior: SpliceResult::CrossedUncovered. The caller
    // keeps its own Arc handle to the prior view, so the engine can
    // preserve Profile.current across the breach; caller emits
    // Diagnostic so the contract violation is observable.
    let mut tree = Tree::new();
    let ids = ensure_chain(&mut tree, &["anchor", "a", "b"]);
    let anchor = ids[0];
    let b = ids[2];

    let mut root_entries = BTreeMap::new();
    root_entries.insert(name("a"), dir(2, 0, None));
    let root = make_dir(meta(1, 1, 0), 0, root_entries);

    let replacement_b = make_dir(meta(3, 3, 0), 0, BTreeMap::new());
    let root_strong_before = Arc::strong_count(&root);
    let s = splice(Some(Arc::clone(&root)), anchor, b, replacement_b, &tree);
    assert!(
        matches!(s, SpliceResult::CrossedUncovered),
        "uncovered intermediate ⇒ CrossedUncovered",
    );
    assert_eq!(
        Arc::strong_count(&root),
        root_strong_before,
        "caller's prior Arc handle survives the splice consume",
    );
}

// ---------------------------------------------------------------------------
// diff_tree
// ---------------------------------------------------------------------------

#[test]
fn diff_tree_self_is_empty() {
    let s = TreeSnapshot::Dir(make_dir(meta(1, 1, 0), 0, BTreeMap::new()));
    let d = diff_tree(&s, &s);
    assert!(d.is_empty());
}

#[test]
fn diff_tree_dir_hash_short_circuit() {
    // Two structurally-equal Dir snapshots must short-circuit and emit
    // an empty Diff regardless of how deep the tree is.
    let inner_a = make_dir(
        meta(2, 2, 0),
        0,
        BTreeMap::from_iter([(
            name("file"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    );
    let inner_b = make_dir(
        meta(2, 2, 0),
        0,
        BTreeMap::from_iter([(
            name("file"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    );
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(name("d"), dir(2, 0, Some(inner_a)))]),
    ));
    let b = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(name("d"), dir(2, 0, Some(inner_b)))]),
    ));
    let d = diff_tree(&a, &b);
    assert!(d.is_empty());
}

#[test]
fn diff_tree_single_leaf_modified() {
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    ));
    let b = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 2, 7, 0)), // mtime bumped
        )]),
    ));
    let d = diff_tree(&a, &b);
    assert_eq!(d.modified.len(), 1);
    assert_eq!(d.modified[0].segment.as_str(), "foo");
    assert_eq!(d.modified[0].kind, EntryKind::File);
    assert!(d.created.is_empty());
    assert!(d.deleted.is_empty());
    assert!(d.renamed.is_empty());
}

#[test]
fn diff_tree_single_leaf_created() {
    let a = TreeSnapshot::Dir(make_dir(meta(1, 1, 0), 0, BTreeMap::new()));
    let b = TreeSnapshot::Dir(make_dir(
        meta(2, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    ));
    let d = diff_tree(&a, &b);
    assert_eq!(d.created.len(), 1);
    assert_eq!(d.created[0].segment.as_str(), "foo");
    assert!(d.deleted.is_empty());
    assert!(d.modified.is_empty());
    assert!(d.renamed.is_empty());
}

#[test]
fn diff_tree_single_leaf_deleted() {
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    ));
    let b = TreeSnapshot::Dir(make_dir(meta(2, 1, 0), 0, BTreeMap::new()));
    let d = diff_tree(&a, &b);
    assert_eq!(d.deleted.len(), 1);
    assert_eq!(d.deleted[0].segment.as_str(), "foo");
    assert!(d.created.is_empty());
}

#[test]
fn diff_tree_single_dir_created_emits_descendants() {
    let inner = make_dir(
        meta(2, 2, 0),
        0,
        BTreeMap::from_iter([(
            name("file"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    );
    let a = TreeSnapshot::Dir(make_dir(meta(1, 1, 0), 0, BTreeMap::new()));
    let b = TreeSnapshot::Dir(make_dir(
        meta(2, 1, 0),
        0,
        BTreeMap::from_iter([(name("d"), dir(2, 0, Some(inner)))]),
    ));
    let d = diff_tree(&a, &b);
    let segs: Vec<_> = d.created.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(
        segs,
        vec!["d", "d/file"],
        "Dir create emits dir + descendants"
    );
    assert!(d.deleted.is_empty());
    assert!(d.renamed.is_empty());
}

#[test]
fn diff_tree_single_dir_deleted_emits_descendants() {
    let inner = make_dir(
        meta(2, 2, 0),
        0,
        BTreeMap::from_iter([(
            name("file"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    );
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(name("d"), dir(2, 0, Some(inner)))]),
    ));
    let b = TreeSnapshot::Dir(make_dir(meta(2, 1, 0), 0, BTreeMap::new()));
    let d = diff_tree(&a, &b);
    let segs: Vec<_> = d.deleted.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(segs, vec!["d", "d/file"]);
}

#[test]
fn diff_tree_same_level_rename() {
    // Same inode at name "foo" in baseline → name "bar" in current.
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    ));
    let b = TreeSnapshot::Dir(make_dir(
        meta(2, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("bar"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    ));
    let d = diff_tree(&a, &b);
    assert_eq!(d.renamed.len(), 1);
    assert_eq!(d.renamed[0].from.segment.as_str(), "foo");
    assert_eq!(d.renamed[0].to.segment.as_str(), "bar");
    assert!(d.created.is_empty());
    assert!(d.deleted.is_empty());
}

#[test]
fn diff_tree_cross_level_rename() {
    // Baseline: /a/foo (inode 7). Current: /b/foo (same inode).
    let a_inner = make_dir(
        meta(2, 2, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    );
    let b_inner = make_dir(
        meta(2, 3, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    );
    let baseline = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(name("a"), dir(2, 0, Some(a_inner)))]),
    ));
    let current = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(name("b"), dir(3, 0, Some(b_inner)))]),
    ));
    let d = diff_tree(&baseline, &current);
    let renames: Vec<(&str, &str)> = d
        .renamed
        .iter()
        .map(|r| (r.from.segment.as_str(), r.to.segment.as_str()))
        .collect();
    // The pair_renames pass should match the inode-7 leaf across levels.
    assert!(
        renames.contains(&("a/foo", "b/foo")),
        "expected /a/foo → /b/foo rename, got {renames:?}",
    );
}

#[test]
fn diff_tree_same_name_different_inode_emits_pair() {
    // Same name, different inode: pair_renames sees same `rel` and skips
    // the rename, leaving Created+Deleted unpaired in their lists.
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    ));
    let b = TreeSnapshot::Dir(make_dir(
        meta(2, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 8, 0)), // new inode
        )]),
    ));
    let d = diff_tree(&a, &b);
    assert!(
        d.renamed.is_empty(),
        "same name + new inode is not a rename"
    );
    assert_eq!(d.deleted.len(), 1);
    assert_eq!(d.created.len(), 1);
    assert_eq!(d.deleted[0].segment.as_str(), "foo");
    assert_eq!(d.created[0].segment.as_str(), "foo");
    assert_eq!(d.deleted[0].fs_id.inode, 7);
    assert_eq!(d.created[0].fs_id.inode, 8);
}

#[test]
fn diff_tree_same_name_kind_change() {
    // Same name "x", file in baseline, dir in current.
    let new_dir = make_dir(meta(2, 8, 0), 0, BTreeMap::new());
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("x"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    ));
    let b = TreeSnapshot::Dir(make_dir(
        meta(2, 1, 0),
        0,
        BTreeMap::from_iter([(name("x"), dir(8, 0, Some(new_dir)))]),
    ));
    let d = diff_tree(&a, &b);
    assert_eq!(d.deleted.len(), 1);
    assert_eq!(d.deleted[0].kind, EntryKind::File);
    assert_eq!(d.created.len(), 1);
    assert_eq!(d.created[0].kind, EntryKind::Dir);
    assert!(d.renamed.is_empty(), "kind change is not a rename");
}

#[test]
fn diff_tree_same_name_kind_change_dir_with_descendants() {
    // Baseline: /foo (File). Current: /foo (Dir with two children).
    // The kind-change arm must recurse into the new Dir's subtree so the
    // descendants surface to the engine's per-stable-file emission path.
    let new_dir = make_dir(
        meta(2, 8, 0),
        0,
        BTreeMap::from_iter([
            (
                name("x"),
                ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 10, 0)),
            ),
            (
                name("y"),
                ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 11, 0)),
            ),
        ]),
    );
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 7, 0)),
        )]),
    ));
    let b = TreeSnapshot::Dir(make_dir(
        meta(2, 1, 0),
        0,
        BTreeMap::from_iter([(name("foo"), dir(8, 0, Some(new_dir)))]),
    ));
    let d = diff_tree(&a, &b);

    let created_segs: Vec<_> = d.created.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(
        created_segs,
        vec!["foo", "foo/x", "foo/y"],
        "kind change → Dir must surface descendants in lex order",
    );
    let deleted_segs: Vec<_> = d.deleted.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(deleted_segs, vec!["foo"]);
    assert!(d.renamed.is_empty());
}

#[test]
fn diff_tree_same_name_kind_change_dir_to_file_with_prior_descendants() {
    // Reverse direction: baseline /foo (Dir with two children),
    // current /foo (File). The prior subtree's descendants must surface
    // as Deleted so consumers see they're gone.
    let prior_dir = make_dir(
        meta(1, 8, 0),
        0,
        BTreeMap::from_iter([
            (
                name("x"),
                ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 10, 0)),
            ),
            (
                name("y"),
                ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 11, 0)),
            ),
        ]),
    );
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(name("foo"), dir(8, 0, Some(prior_dir)))]),
    ));
    let b = TreeSnapshot::Dir(make_dir(
        meta(2, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 2, 7, 0)),
        )]),
    ));
    let d = diff_tree(&a, &b);

    let deleted_segs: Vec<_> = d.deleted.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(
        deleted_segs,
        vec!["foo", "foo/x", "foo/y"],
        "kind change → Leaf must surface prior descendants in lex order",
    );
    let created_segs: Vec<_> = d.created.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(created_segs, vec!["foo"]);
    assert!(d.renamed.is_empty());
}

#[test]
fn diff_tree_same_name_kind_change_with_inode_collision() {
    // Both prior and new at "foo" share inode 100, but kind differs
    // (Dir → File via kernel inode reuse). The rename pairing layer must
    // NOT collapse them into a (nonsensical) same-name "Rename".
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            dir(100, 0, Some(make_dir(meta(1, 100, 0), 0, BTreeMap::new()))),
        )]),
    ));
    let b = TreeSnapshot::Dir(make_dir(
        meta(2, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("foo"),
            ChildEntry::Leaf(leaf(EntryKind::File, 0, 0, 100, 0)),
        )]),
    ));
    let d = diff_tree(&a, &b);
    assert!(
        d.renamed.is_empty(),
        "same-path kind change with inode reuse is not a rename, got {:?}",
        d.renamed,
    );
    assert_eq!(d.deleted.len(), 1);
    assert_eq!(d.deleted[0].kind, EntryKind::Dir);
    assert_eq!(d.deleted[0].fs_id.inode, 100);
    assert_eq!(d.created.len(), 1);
    assert_eq!(d.created[0].kind, EntryKind::File);
    assert_eq!(d.created[0].fs_id.inode, 100);
}

#[test]
fn diff_tree_cross_kind_inode_collision_no_phantom_rename() {
    // Different paths, different kinds, same inode (kernel reuse across
    // unrelated rm + mkdir). Must NOT be paired as a "Rename file → dir".
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("alpha"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 100, 0)),
        )]),
    ));
    let b = TreeSnapshot::Dir(make_dir(
        meta(2, 1, 0),
        0,
        BTreeMap::from_iter([(
            name("beta"),
            dir(100, 0, Some(make_dir(meta(2, 100, 0), 0, BTreeMap::new()))),
        )]),
    ));
    let d = diff_tree(&a, &b);
    assert!(
        d.renamed.is_empty(),
        "cross-kind inode collision is not a rename, got {:?}",
        d.renamed,
    );
    let deleted_segs: Vec<_> = d.deleted.iter().map(|e| e.segment.as_str()).collect();
    let created_segs: Vec<_> = d.created.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(deleted_segs, vec!["alpha"]);
    assert_eq!(created_segs, vec!["beta"]);
    assert_eq!(d.deleted[0].kind, EntryKind::File);
    assert_eq!(d.created[0].kind, EntryKind::Dir);
}

#[test]
fn diff_tree_dir_replace_at_different_inode_emits_descendants() {
    // Same-name dir replaced with a fresh dir at a different inode (the
    // user `rm -rf foo && mkdir foo`). Both the parent slot and every
    // descendant on each side must surface — the parent identity is
    // structurally different, and the prior children are gone while the
    // new children are new.
    let prior_inner = make_dir(
        meta(1, 200, 0),
        0,
        BTreeMap::from_iter([
            (
                name("x"),
                ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 10, 0)),
            ),
            (
                name("y"),
                ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 11, 0)),
            ),
        ]),
    );
    let new_inner = make_dir(
        meta(2, 300, 0),
        0,
        BTreeMap::from_iter([
            (
                name("p"),
                ChildEntry::Leaf(leaf(EntryKind::File, 1, 2, 20, 0)),
            ),
            (
                name("q"),
                ChildEntry::Leaf(leaf(EntryKind::File, 1, 2, 21, 0)),
            ),
        ]),
    );
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([(name("foo"), dir(200, 0, Some(prior_inner)))]),
    ));
    let b = TreeSnapshot::Dir(make_dir(
        meta(2, 1, 0),
        0,
        BTreeMap::from_iter([(name("foo"), dir(300, 0, Some(new_inner)))]),
    ));
    let d = diff_tree(&a, &b);

    let deleted_segs: Vec<_> = d.deleted.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(
        deleted_segs,
        vec!["foo", "foo/x", "foo/y"],
        "dir-replace must surface prior descendants in lex order",
    );
    let created_segs: Vec<_> = d.created.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(
        created_segs,
        vec!["foo", "foo/p", "foo/q"],
        "dir-replace must surface new descendants in lex order",
    );
    assert!(d.renamed.is_empty());
}

#[test]
fn diff_tree_rename_into_kind_change_slot() {
    // Combined scenario: /old changes kind File→Dir AND a file at
    // /something/dir moves into the new Dir as /old/x (same inode 500).
    // The kind change must surface as Deleted+Created at /old, and the
    // genuine cross-level move must surface as a Rename — neither should
    // be lost or collapsed by the staging architecture.
    let new_inner = make_dir(
        meta(2, 200, 0),
        0,
        BTreeMap::from_iter([(
            name("x"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 2, 500, 0)),
        )]),
    );
    let prior_something = make_dir(
        meta(1, 999, 0),
        0,
        BTreeMap::from_iter([(
            name("dir"),
            ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 500, 0)),
        )]),
    );
    let new_something = make_dir(meta(2, 999, 0), 0, BTreeMap::new());
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([
            (
                name("old"),
                ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 100, 0)),
            ),
            (name("something"), dir(999, 0, Some(prior_something))),
        ]),
    ));
    let b = TreeSnapshot::Dir(make_dir(
        meta(2, 1, 0),
        0,
        BTreeMap::from_iter([
            (name("old"), dir(200, 0, Some(new_inner))),
            (name("something"), dir(999, 0, Some(new_something))),
        ]),
    ));
    let d = diff_tree(&a, &b);

    let renames: Vec<(&str, &str)> = d
        .renamed
        .iter()
        .map(|r| (r.from.segment.as_str(), r.to.segment.as_str()))
        .collect();
    assert_eq!(
        renames,
        vec![("something/dir", "old/x")],
        "cross-level rename into kind-changed slot must be detected",
    );
    let deleted_segs: Vec<_> = d.deleted.iter().map(|e| e.segment.as_str()).collect();
    let created_segs: Vec<_> = d.created.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(
        deleted_segs,
        vec!["old"],
        "kind change at /old must surface as Deleted (File)",
    );
    assert_eq!(
        created_segs,
        vec!["old"],
        "kind change at /old must surface as Created (Dir); /old/x was paired as Rename",
    );
    assert_eq!(d.deleted[0].kind, EntryKind::File);
    assert_eq!(d.deleted[0].fs_id.inode, 100);
    assert_eq!(d.created[0].kind, EntryKind::Dir);
    assert_eq!(d.created[0].fs_id.inode, 200);
}

#[test]
fn diff_tree_file_pair_device_change_is_delete_create() {
    // File-anchored Profile, anchor's device flips. `diff_file_pair` must
    // emit Deleted+Created (not Modified): the snapshot identity changed.
    let a = TreeSnapshot::File(leaf(EntryKind::File, 10, 1, 7, 0));
    let b = TreeSnapshot::File(leaf(EntryKind::File, 10, 1, 7, 1));
    let d = diff_tree(&a, &b);
    assert_eq!(d.deleted.len(), 1);
    assert_eq!(d.created.len(), 1);
    assert_eq!(d.deleted[0].fs_id.inode, 7);
    assert_eq!(d.created[0].fs_id.inode, 7);
    assert!(d.modified.is_empty());
    assert!(d.renamed.is_empty());
}

#[test]
fn diff_tree_modified_lists_in_lex_order() {
    let a = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([
            (
                name("a_first"),
                ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 1, 0)),
            ),
            (
                name("z_last"),
                ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 2, 0)),
            ),
            (
                name("m_mid"),
                ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 3, 0)),
            ),
        ]),
    ));
    let b = TreeSnapshot::Dir(make_dir(
        meta(1, 1, 0),
        0,
        BTreeMap::from_iter([
            (
                name("a_first"),
                ChildEntry::Leaf(leaf(EntryKind::File, 10, 9, 1, 0)),
            ),
            (
                name("z_last"),
                ChildEntry::Leaf(leaf(EntryKind::File, 10, 9, 2, 0)),
            ),
            (
                name("m_mid"),
                ChildEntry::Leaf(leaf(EntryKind::File, 10, 9, 3, 0)),
            ),
        ]),
    ));
    let d = diff_tree(&a, &b);
    let segs: Vec<_> = d.modified.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(segs, vec!["a_first", "m_mid", "z_last"]);
}

#[test]
fn diff_tree_created_lists_in_lex_order() {
    let a = TreeSnapshot::Dir(make_dir(meta(1, 1, 0), 0, BTreeMap::new()));
    let b = TreeSnapshot::Dir(make_dir(
        meta(2, 1, 0),
        0,
        BTreeMap::from_iter([
            (
                name("z_last"),
                ChildEntry::Leaf(leaf(EntryKind::File, 0, 0, 3, 0)),
            ),
            (
                name("a_first"),
                ChildEntry::Leaf(leaf(EntryKind::File, 0, 0, 1, 0)),
            ),
            (
                name("m_mid"),
                ChildEntry::Leaf(leaf(EntryKind::File, 0, 0, 2, 0)),
            ),
        ]),
    ));
    let d = diff_tree(&a, &b);
    let segs: Vec<_> = d.created.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(segs, vec!["a_first", "m_mid", "z_last"]);
}

#[test]
fn diff_tree_file_to_file_modified() {
    let a = TreeSnapshot::File(leaf(EntryKind::File, 10, 1, 7, 0));
    let b = TreeSnapshot::File(leaf(EntryKind::File, 10, 2, 7, 0)); // mtime bumped
    let d = diff_tree(&a, &b);
    assert_eq!(d.modified.len(), 1);
    assert_eq!(d.modified[0].segment.as_str(), "");
    assert_eq!(d.modified[0].fs_id.inode, 7);
}

#[test]
fn diff_tree_file_to_file_inode_change() {
    let a = TreeSnapshot::File(leaf(EntryKind::File, 10, 1, 7, 0));
    let b = TreeSnapshot::File(leaf(EntryKind::File, 10, 1, 8, 0));
    let d = diff_tree(&a, &b);
    assert_eq!(d.deleted.len(), 1);
    assert_eq!(d.created.len(), 1);
    assert_eq!(d.deleted[0].fs_id.inode, 7);
    assert_eq!(d.created[0].fs_id.inode, 8);
    assert!(
        d.renamed.is_empty(),
        "File-anchor inode flip is delete+create, not rename",
    );
}

#[test]
fn diff_tree_recursive_three_levels_deep_change() {
    // anchor → a → b: only b's contents differ. dir_hash short-circuits
    // at any unchanged sibling but recurses through a → b until the
    // affected leaf.
    fn build(top_mtime: u64, leaf_mtime: u64) -> TreeSnapshot {
        let b = make_dir(
            meta(3, 3, 0),
            0,
            BTreeMap::from_iter([(
                name("file"),
                ChildEntry::Leaf(leaf(EntryKind::File, 10, leaf_mtime, 7, 0)),
            )]),
        );
        let a = make_dir(
            meta(2, 2, 0),
            0,
            BTreeMap::from_iter([(name("b"), dir(3, 0, Some(b)))]),
        );
        let other = make_dir(
            meta(2, 99, 0),
            0,
            BTreeMap::from_iter([(
                name("untouched"),
                ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 1, 0)),
            )]),
        );
        TreeSnapshot::Dir(make_dir(
            meta(top_mtime, 1, 0),
            0,
            BTreeMap::from_iter([
                (name("a"), dir(2, 0, Some(a))),
                (name("other"), dir(99, 0, Some(other))),
            ]),
        ))
    }
    let baseline = build(1, 1);
    let current = build(1, 2); // leaf mtime bumped — only one change

    // Note: the `other` subtree has matching dir_hash across baseline and
    // current; we expect short-circuit at that sibling.
    let d = diff_tree(&baseline, &current);
    assert_eq!(d.modified.len(), 1);
    assert_eq!(d.modified[0].segment.as_str(), "a/b/file");
    assert!(d.created.is_empty());
    assert!(d.deleted.is_empty());
    assert!(d.renamed.is_empty());
}

// ---------------------------------------------------------------------------
// Diff::all_created / Diff::all_deleted
// ---------------------------------------------------------------------------

#[test]
fn diff_all_created_empty_snapshot_yields_empty_diff() {
    let snap = DirSnapshot::new(meta(1, 100, 0), 0, BTreeMap::new());
    let d = Diff::all_created(&snap);
    assert!(d.is_empty());
}

#[test]
fn diff_all_deleted_empty_snapshot_yields_empty_diff() {
    let snap = DirSnapshot::new(meta(1, 100, 0), 0, BTreeMap::new());
    let d = Diff::all_deleted(&snap);
    assert!(d.is_empty());
}

#[test]
fn diff_all_created_single_leaf_emits_one_created_entry() {
    let mut entries = BTreeMap::new();
    entries.insert(
        name("a.rs"),
        ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 42, 0)),
    );
    let snap = DirSnapshot::new(meta(1, 100, 0), 0, entries);

    let d = Diff::all_created(&snap);
    assert_eq!(d.created.len(), 1);
    assert_eq!(d.created[0].segment.as_str(), "a.rs");
    assert_eq!(d.created[0].kind, EntryKind::File);
    assert_eq!(d.created[0].fs_id.inode, 42);
    assert!(d.deleted.is_empty());
    assert!(d.modified.is_empty());
    assert!(d.renamed.is_empty());
}

#[test]
fn diff_all_deleted_single_leaf_emits_one_deleted_entry() {
    let mut entries = BTreeMap::new();
    entries.insert(
        name("a.rs"),
        ChildEntry::Leaf(leaf(EntryKind::File, 10, 1, 42, 0)),
    );
    let snap = DirSnapshot::new(meta(1, 100, 0), 0, entries);

    let d = Diff::all_deleted(&snap);
    assert_eq!(d.deleted.len(), 1);
    assert_eq!(d.deleted[0].segment.as_str(), "a.rs");
    assert_eq!(d.deleted[0].kind, EntryKind::File);
    assert_eq!(d.deleted[0].fs_id.inode, 42);
    assert!(d.created.is_empty());
    assert!(d.modified.is_empty());
    assert!(d.renamed.is_empty());
}

#[test]
fn diff_all_created_uncovered_dir_emits_dir_only_no_descendants() {
    // `DirChild::Uncovered(_)` — the walker stored the entry but
    // didn't recurse. all_created emits the Dir entry itself but
    // cannot synthesise descendants it never saw.
    let mut entries = BTreeMap::new();
    entries.insert(name("sub"), dir(7, 0, None));
    let snap = DirSnapshot::new(meta(1, 100, 0), 0, entries);

    let d = Diff::all_created(&snap);
    assert_eq!(d.created.len(), 1);
    assert_eq!(d.created[0].segment.as_str(), "sub");
    assert_eq!(d.created[0].kind, EntryKind::Dir);
    assert_eq!(d.created[0].fs_id.inode, 7);
}

#[test]
fn diff_all_created_covered_dir_emits_recursive_entries_depth_first_lex() {
    // Structure: root/{a (Dir, covered)/{b.rs}, c.rs}.
    // Expected depth-first lex order in `created`:
    //   "a", "a/b.rs", "c.rs".
    let mut inner = BTreeMap::new();
    inner.insert(
        name("b.rs"),
        ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 11, 0)),
    );
    let inner_snap = make_dir(meta(1, 7, 0), 0, inner);

    let mut entries = BTreeMap::new();
    entries.insert(name("a"), dir(7, 0, Some(Arc::clone(&inner_snap))));
    entries.insert(
        name("c.rs"),
        ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 22, 0)),
    );
    let snap = DirSnapshot::new(meta(1, 100, 0), 0, entries);

    let d = Diff::all_created(&snap);
    let segments: Vec<&str> = d.created.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(segments, vec!["a", "a/b.rs", "c.rs"]);
}

#[test]
fn diff_all_deleted_covered_dir_emits_recursive_entries_depth_first_lex() {
    // Symmetric to `diff_all_created_covered_dir_emits_recursive_entries_depth_first_lex`.
    let mut inner = BTreeMap::new();
    inner.insert(
        name("b.rs"),
        ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 11, 0)),
    );
    let inner_snap = make_dir(meta(1, 7, 0), 0, inner);

    let mut entries = BTreeMap::new();
    entries.insert(name("a"), dir(7, 0, Some(Arc::clone(&inner_snap))));
    entries.insert(
        name("c.rs"),
        ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 22, 0)),
    );
    let snap = DirSnapshot::new(meta(1, 100, 0), 0, entries);

    let d = Diff::all_deleted(&snap);
    let segments: Vec<&str> = d.deleted.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(segments, vec!["a", "a/b.rs", "c.rs"]);
}

#[test]
fn diff_all_created_deep_nesting() {
    // Three-level deep nesting: r/{x (Dir)/{y (Dir)/{leaf.rs}}}.
    let mut l3 = BTreeMap::new();
    l3.insert(
        name("leaf.rs"),
        ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 13, 0)),
    );
    let l3_snap = make_dir(meta(1, 12, 0), 0, l3);

    let mut l2 = BTreeMap::new();
    l2.insert(name("y"), dir(12, 0, Some(l3_snap)));
    let l2_snap = make_dir(meta(1, 11, 0), 0, l2);

    let mut l1 = BTreeMap::new();
    l1.insert(name("x"), dir(11, 0, Some(l2_snap)));
    let snap = DirSnapshot::new(meta(1, 10, 0), 0, l1);

    let d = Diff::all_created(&snap);
    let segments: Vec<&str> = d.created.iter().map(|e| e.segment.as_str()).collect();
    assert_eq!(segments, vec!["x", "x/y", "x/y/leaf.rs"]);
}

#[test]
fn diff_all_created_matches_diff_against_empty_baseline() {
    // Equivalence with the canonical form: diff_tree(empty, snap).created
    // should equal Diff::all_created(snap).created (segment + kind + inode
    // tuples, ignoring SmallVec capacity differences).
    let mut inner = BTreeMap::new();
    inner.insert(
        name("file.rs"),
        ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 11, 0)),
    );
    let inner_snap = make_dir(meta(1, 7, 0), 0, inner);

    let mut entries = BTreeMap::new();
    entries.insert(name("sub"), dir(7, 0, Some(inner_snap)));
    entries.insert(
        name("top.rs"),
        ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 22, 0)),
    );
    let snap = make_dir(meta(1, 100, 0), 0, entries);

    let empty = make_dir(meta(1, 100, 0), 0, BTreeMap::new());
    let canonical = diff_tree(
        &TreeSnapshot::Dir(empty),
        &TreeSnapshot::Dir(Arc::clone(&snap)),
    );
    let shorthand = Diff::all_created(&snap);

    let canon_tuples: Vec<(String, EntryKind, u64)> = canonical
        .created
        .iter()
        .map(|e| (e.segment.to_string(), e.kind, e.fs_id.inode))
        .collect();
    let short_tuples: Vec<(String, EntryKind, u64)> = shorthand
        .created
        .iter()
        .map(|e| (e.segment.to_string(), e.kind, e.fs_id.inode))
        .collect();
    assert_eq!(canon_tuples, short_tuples);

    // Modified/renamed/deleted must be empty on both sides.
    assert!(canonical.deleted.is_empty());
    assert!(canonical.modified.is_empty());
    assert!(canonical.renamed.is_empty());
    assert!(shorthand.deleted.is_empty());
    assert!(shorthand.modified.is_empty());
    assert!(shorthand.renamed.is_empty());
}

#[test]
fn diff_all_deleted_matches_diff_from_empty_target() {
    // Symmetric equivalence: diff_tree(snap, empty).deleted should match
    // Diff::all_deleted(snap).deleted.
    let mut entries = BTreeMap::new();
    entries.insert(
        name("top.rs"),
        ChildEntry::Leaf(leaf(EntryKind::File, 1, 1, 22, 0)),
    );
    let snap = make_dir(meta(1, 100, 0), 0, entries);
    let empty = make_dir(meta(1, 100, 0), 0, BTreeMap::new());

    let canonical = diff_tree(
        &TreeSnapshot::Dir(Arc::clone(&snap)),
        &TreeSnapshot::Dir(empty),
    );
    let shorthand = Diff::all_deleted(&snap);

    let canon_tuples: Vec<(String, EntryKind, u64)> = canonical
        .deleted
        .iter()
        .map(|e| (e.segment.to_string(), e.kind, e.fs_id.inode))
        .collect();
    let short_tuples: Vec<(String, EntryKind, u64)> = shorthand
        .deleted
        .iter()
        .map(|e| (e.segment.to_string(), e.kind, e.fs_id.inode))
        .collect();
    assert_eq!(canon_tuples, short_tuples);
}

// ---------------------------------------------------------------------------
// TreeSnapshot::hash
// ---------------------------------------------------------------------------

#[test]
fn tree_snapshot_hash_returns_dir_hash_for_dir_variant() {
    let arc = make_dir(meta(1, 100, 1), 0, BTreeMap::new());
    let snap = TreeSnapshot::Dir(Arc::clone(&arc));
    assert_eq!(snap.hash(), arc.dir_hash());
}

#[test]
fn tree_snapshot_hash_returns_leaf_hash_for_file_variant() {
    let l = leaf(EntryKind::File, 10, 1, 42, 0);
    let snap = TreeSnapshot::File(l.clone());
    assert_eq!(snap.hash(), l.leaf_hash());
}

// ---------------------------------------------------------------------------
// Property tests
// ---------------------------------------------------------------------------

fn arb_kind() -> impl Strategy<Value = EntryKind> {
    prop_oneof![
        Just(EntryKind::File),
        Just(EntryKind::Symlink),
        Just(EntryKind::Other),
    ]
}

fn arb_leaf() -> impl Strategy<Value = LeafEntry> {
    (arb_kind(), 0u64..1024, 0u64..1024, 1u64..1024, 0u64..4)
        .prop_map(|(k, sz, mt, ino, dev)| leaf(k, sz, mt, ino, dev))
}

fn arb_simple_entries() -> impl Strategy<Value = BTreeMap<CompactString, ChildEntry>> {
    proptest::collection::vec(("[a-z]{1,4}", arb_leaf()), 0..6).prop_map(|v| {
        let mut m = BTreeMap::new();
        for (i, (s, l)) in v.into_iter().enumerate() {
            // Disambiguate: BTreeMap drops duplicates; the proptest may
            // generate the same name twice. Index-prefix to keep the name
            // unique while preserving lex sortability.
            m.insert(CompactString::new(format!("{i}_{s}")), ChildEntry::Leaf(l));
        }
        m
    })
}

proptest! {
    /// Same inputs ⇒ same dir_hash, regardless of insertion order
    /// (BTreeMap is sorted-by-key, but two separate constructions with
    /// the same data must agree).
    #[test]
    fn prop_dir_hash_deterministic(
        meta_secs in 0u64..100,
        meta_inode in 1u64..1000,
        captured_with in 0u64..16,
        e in arb_simple_entries(),
    ) {
        let m = meta(meta_secs, meta_inode, 0);
        let a = make_dir(m, captured_with, e.clone());
        let b = make_dir(m, captured_with, e);
        prop_assert_eq!(a.dir_hash(), b.dir_hash());
    }

    /// Same inputs in any insertion order ⇒ same hash. BTreeMap sorts by
    /// key, so iteration order is deterministic regardless of insertion
    /// order. Belt-and-suspenders: verify via reverse-order rebuild.
    #[test]
    fn prop_dir_hash_order_independent(
        e in arb_simple_entries(),
    ) {
        let m = meta(1, 1, 0);
        let a = make_dir(m, 0, e.clone());
        let mut reversed: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
        for (k, v) in e.into_iter().rev() {
            reversed.insert(k, v);
        }
        let b = make_dir(m, 0, reversed);
        prop_assert_eq!(a.dir_hash(), b.dir_hash());
    }

    #[test]
    fn prop_leaf_hash_deterministic(l in arb_leaf()) {
        let a = l.clone();
        let b = l;
        prop_assert_eq!(a.leaf_hash(), b.leaf_hash());
    }

    #[test]
    fn prop_dir_hash_distinguishes_root_meta_inode_field(
        e in arb_simple_entries(),
    ) {
        let a = make_dir(meta(1, 100, 0), 0, e.clone());
        let b = make_dir(meta(1, 101, 0), 0, e);
        prop_assert_ne!(a.dir_hash(), b.dir_hash());
    }

    #[test]
    fn prop_diff_tree_self_is_empty(e in arb_simple_entries()) {
        let s = TreeSnapshot::Dir(make_dir(meta(1, 1, 0), 0, e));
        let d = diff_tree(&s, &s);
        prop_assert!(d.is_empty());
    }

    /// Inverse symmetry: diff(a,b).created == diff(b,a).deleted (as
    /// segment-and-kind sets, ignoring renames). Renames flip from↔to.
    #[test]
    fn prop_diff_tree_inverse(
        ea in arb_simple_entries(),
        eb in arb_simple_entries(),
    ) {
        let a = TreeSnapshot::Dir(make_dir(meta(1, 1, 0), 0, ea));
        let b = TreeSnapshot::Dir(make_dir(meta(2, 1, 0), 0, eb));
        let fwd = diff_tree(&a, &b);
        let rev = diff_tree(&b, &a);

        let fwd_created: std::collections::BTreeSet<(String, EntryKind, u64)> = fwd
            .created
            .iter()
            .map(|e| (e.segment.to_string(), e.kind, e.fs_id.inode))
            .collect();
        let rev_deleted: std::collections::BTreeSet<(String, EntryKind, u64)> = rev
            .deleted
            .iter()
            .map(|e| (e.segment.to_string(), e.kind, e.fs_id.inode))
            .collect();
        prop_assert_eq!(fwd_created, rev_deleted);

        let fwd_deleted: std::collections::BTreeSet<(String, EntryKind, u64)> = fwd
            .deleted
            .iter()
            .map(|e| (e.segment.to_string(), e.kind, e.fs_id.inode))
            .collect();
        let rev_created: std::collections::BTreeSet<(String, EntryKind, u64)> = rev
            .created
            .iter()
            .map(|e| (e.segment.to_string(), e.kind, e.fs_id.inode))
            .collect();
        prop_assert_eq!(fwd_deleted, rev_created);
    }

    /// Off-path Arc preservation: splice at a single subtree leaves the
    /// other top-level children Arc::ptr_eq with their pre-splice values.
    #[test]
    fn prop_splice_off_path_unchanged(
        leaf_size in 1u64..10,
        sibling_count in 0usize..4,
    ) {
        let mut tree = Tree::new();
        let ids = ensure_chain(&mut tree, &["anchor", "a"]);
        let anchor = ids[0];
        let a = ids[1];

        let prior_a = make_dir(meta(2, 2, 0), 0, BTreeMap::new());
        let mut root_entries = BTreeMap::new();
        root_entries.insert(name("a"), dir(2, 0, Some(Arc::clone(&prior_a))));
        let mut siblings: Vec<Arc<DirSnapshot>> = Vec::new();
        for i in 0..sibling_count {
            let sib_name = format!("sib_{i}");
            let _sib_id = tree.ensure(Some(anchor), &sib_name, ResourceRole::User);
            let sib = make_dir(meta(50 + i as u64, 50 + i as u64, 0), 0, BTreeMap::new());
            siblings.push(Arc::clone(&sib));
            root_entries.insert(CompactString::new(sib_name), dir(50 + i as u64, 0, Some(sib)));
        }
        let root = make_dir(meta(1, 1, 0), 0, root_entries);

        // Replacement carries a child whose size is proptest-driven; this
        // yields a structurally distinct snapshot whose dir_hash differs
        // from the empty prior_a regardless of `root_meta.mtime`.
        let mut replacement_entries = BTreeMap::new();
        replacement_entries.insert(
            name("file"),
            ChildEntry::Leaf(leaf(EntryKind::File, leaf_size, 1, 7, 0)),
        );
        let replacement = make_dir(meta(2, 2, 0), 0, replacement_entries);
        prop_assert_ne!(prior_a.dir_hash(), replacement.dir_hash());
        let s = splice(Some(root), anchor, a, replacement, &tree);
        let SpliceResult::Spliced(TreeSnapshot::Dir(new_root)) = s else { unreachable!() };
        for (i, sib) in siblings.iter().enumerate() {
            let key = format!("sib_{i}");
            let after = dir_subtree(new_root.entries.get(key.as_str()).unwrap());
            prop_assert!(Arc::ptr_eq(after, sib));
        }
    }

    /// splice(prior, anchor, target, replacement) ⇒ subtree_at(target) == replacement.
    #[test]
    fn prop_splice_then_subtree_at_returns_replacement(
        leaf_size in 1u64..50,
    ) {
        let mut tree = Tree::new();
        let ids = ensure_chain(&mut tree, &["anchor", "a", "b"]);
        let anchor = ids[0];
        let b = ids[2];

        let prior_b = make_dir(meta(3, 3, 0), 0, BTreeMap::new());
        let a_snap = make_dir(
            meta(2, 2, 0),
            0,
            BTreeMap::from_iter([(name("b"), dir(3, 0, Some(Arc::clone(&prior_b))))]),
        );
        let root = make_dir(
            meta(1, 1, 0),
            0,
            BTreeMap::from_iter([(name("a"), dir(2, 0, Some(a_snap)))]),
        );
        // Replacement carries a structurally-distinct entry (size is
        // proptest-driven); under filter-aware `dir_hash` this guarantees
        // a hash difference regardless of `root_meta.mtime`.
        let mut replacement_entries = BTreeMap::new();
        replacement_entries.insert(
            name("file"),
            ChildEntry::Leaf(leaf(EntryKind::File, leaf_size, 1, 7, 0)),
        );
        let replacement = make_dir(meta(3, 3, 0), 0, replacement_entries);
        prop_assert_ne!(prior_b.dir_hash(), replacement.dir_hash());
        let s = splice(Some(root), anchor, b, Arc::clone(&replacement), &tree);
        let SpliceResult::Spliced(snap) = s else { unreachable!() };
        let got = snap
            .subtree_at(anchor, b, &tree)
            .expect("b resolves after splice");
        prop_assert_eq!(got.dir_hash(), replacement.dir_hash());
    }

    /// G7-trivial property: if replacement.dir_hash() == prior.subtree_at(target).dir_hash(),
    /// splice returns the prior unchanged (Arc::ptr_eq with the input root).
    #[test]
    fn prop_splice_idempotent_with_equal_dir_hash(
        meta_secs in 1u64..50,
    ) {
        let mut tree = Tree::new();
        let ids = ensure_chain(&mut tree, &["anchor", "a"]);
        let anchor = ids[0];
        let a = ids[1];
        let m = meta(meta_secs, 2, 0);
        let prior_a = make_dir(m, 0, BTreeMap::new());
        let root = make_dir(
            meta(1, 1, 0),
            0,
            BTreeMap::from_iter([(name("a"), dir(2, 0, Some(Arc::clone(&prior_a))))]),
        );
        // Equal hash because all data fields agree.
        let replacement = make_dir(m, 0, BTreeMap::new());
        prop_assert_eq!(prior_a.dir_hash(), replacement.dir_hash());
        let s = splice(Some(Arc::clone(&root)), anchor, a, replacement, &tree);
        let SpliceResult::Spliced(TreeSnapshot::Dir(new_root)) = s else { unreachable!() };
        prop_assert!(Arc::ptr_eq(&new_root, &root));
    }
}
