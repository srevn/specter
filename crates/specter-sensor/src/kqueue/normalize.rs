//! kqueue `(flags, fflags, kind)` → `FsEvent` mapping.
//!
//! A single `kevent` may carry multiple NOTE flags (e.g.,
//! `NOTE_WRITE | NOTE_EXTEND | NOTE_ATTRIB` on a file truncate + chmod).
//! We emit at most one `FsEvent` per kevent by priority order:
//!
//! ```text
//! Revoked > Removed > Renamed > <NOTE_LINK kind-aware> > StructureChanged > Modified > MetadataChanged
//! ```
//!
//! The terminal flags (`NOTE_REVOKE` / `NOTE_DELETE` / `NOTE_RENAME`) are
//! exclusive — once one fires, no further events arrive on the fd. Pairing
//! a non-terminal flag with a terminal one (the kernel can do this when
//! flags coalesce) reports the terminal: emitting both would be noise the
//! engine doesn't act on. The non-terminal `Modified` / `MetadataChanged`
//! coalescing is acceptable because the engine's `Settling` state debounces
//! either as "something changed; reschedule."
//!
//! `NOTE_LINK` is kind-aware: on a Dir it's a structural signal
//! (subdirectory created/removed via `..` backref count change); on a
//! File it's a metadata signal (hardlink count change via `ln`/`unlink`).
//! Placed before WRITE/EXTEND in the priority order so a coalesced
//! `(LINK | WRITE)` kevent on a File maps to `MetadataChanged` rather
//! than `Modified`. On a Dir both branches collapse to
//! `StructureChanged` so ordering is observationally irrelevant — but
//! placing LINK first makes the structural intent explicit.
//!
//! `flags` carries `EV_*` bits (e.g., `EV_EOF` from the kernel's
//! auto-detach signal); v1 ignores them. The terminal `FsEvent` is the
//! only signal the engine acts on, and `EV_EOF` always coincides with a
//! terminal `fflag`.

use libc::{
    NOTE_ATTRIB, NOTE_DELETE, NOTE_EXTEND, NOTE_LINK, NOTE_RENAME, NOTE_REVOKE, NOTE_WRITE,
};
use specter_core::{FsEvent, ResourceKind};

/// Map a single kevent's flags into at most one `FsEvent`. Returns
/// `None` if the event carries no actionable signal (e.g., `EV_EOF`
/// alone, or registration acks where no NOTE bit is set).
pub(super) const fn kevent_to_fs_event(
    _flags: u16,
    fflags: u32,
    kind: ResourceKind,
) -> Option<FsEvent> {
    // Terminal flags first — if any fires, that's the only thing the
    // engine cares about.
    if fflags & NOTE_REVOKE != 0 {
        return Some(FsEvent::Revoked);
    }
    if fflags & NOTE_DELETE != 0 {
        return Some(FsEvent::Removed);
    }
    if fflags & NOTE_RENAME != 0 {
        return Some(FsEvent::Renamed);
    }

    // Resolve Unknown→File once via the canonical collapse (the kind
    // cache may transiently desync; defensive folding matches the
    // translator's mask decision so registration and event-shape stay
    // consistent).
    let effective = kind.effective();

    // NOTE_LINK is kind-aware. On a Dir, link-count changes via
    // child-dir creation / removal (the parent's `..` backref count
    // shifts); the engine treats that as `StructureChanged`. On a File,
    // NOTE_LINK fires on hardlink ops (`ln`, `unlink` on a hardlinked
    // inode) — a metadata signal under the class taxonomy.
    //
    // Placed before the WRITE/EXTEND arm so a coalesced `(LINK | WRITE)`
    // kevent on a File yields MetadataChanged (LINK semantics dominate
    // the hardlink-count interpretation). On a Dir both branches yield
    // StructureChanged so ordering is observationally irrelevant — the
    // explicit ordering documents intent.
    if fflags & NOTE_LINK != 0 {
        return Some(if matches!(effective, ResourceKind::Dir) {
            FsEvent::StructureChanged
        } else {
            FsEvent::MetadataChanged
        });
    }

    // Non-terminal: WRITE / EXTEND collapse based on resource kind. A
    // dir's NOTE_WRITE is "an entry in this dir changed" — the engine
    // treats that as `StructureChanged` (probe the dir, diff against
    // current). A file's NOTE_WRITE is `Modified` (file content
    // changed; probe the file's kind/size/mtime).
    if fflags & (NOTE_WRITE | NOTE_EXTEND) != 0 {
        return Some(if matches!(effective, ResourceKind::Dir) {
            FsEvent::StructureChanged
        } else {
            FsEvent::Modified
        });
    }

    if fflags & NOTE_ATTRIB != 0 {
        return Some(FsEvent::MetadataChanged);
    }

    None
}
