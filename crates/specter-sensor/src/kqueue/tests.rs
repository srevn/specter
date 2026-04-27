//! Pure-Rust unit tests for the kqueue submodule. No real kqueue —
//! these exercise the normalization table, the deadline math, and
//! sanity invariants of the wake-ident and ffi conversions.

use super::ffi;
use super::normalize::kevent_to_fs_event;
use super::watcher::deadline_instant_to_timespec;
use specter_core::{FsEvent, ResourceKind};
use std::time::{Duration, Instant};

// ── normalization table ──────────────────────────────────────────────

#[test]
fn normalize_revoke_takes_priority() {
    let fflags = libc::NOTE_REVOKE | libc::NOTE_DELETE | libc::NOTE_WRITE;
    assert_eq!(
        kevent_to_fs_event(0, fflags, ResourceKind::File),
        Some(FsEvent::Revoked)
    );
}

#[test]
fn normalize_remove_takes_priority_over_rename() {
    let fflags = libc::NOTE_DELETE | libc::NOTE_RENAME;
    assert_eq!(
        kevent_to_fs_event(0, fflags, ResourceKind::File),
        Some(FsEvent::Removed)
    );
}

#[test]
fn normalize_rename_takes_priority_over_write() {
    let fflags = libc::NOTE_RENAME | libc::NOTE_WRITE;
    assert_eq!(
        kevent_to_fs_event(0, fflags, ResourceKind::File),
        Some(FsEvent::Renamed)
    );
}

#[test]
fn normalize_write_on_dir_is_structure_changed() {
    assert_eq!(
        kevent_to_fs_event(0, libc::NOTE_WRITE, ResourceKind::Dir),
        Some(FsEvent::StructureChanged)
    );
}

#[test]
fn normalize_write_on_file_is_modified() {
    assert_eq!(
        kevent_to_fs_event(0, libc::NOTE_WRITE, ResourceKind::File),
        Some(FsEvent::Modified)
    );
}

#[test]
fn normalize_extend_alone_collapses_with_write() {
    assert_eq!(
        kevent_to_fs_event(0, libc::NOTE_EXTEND, ResourceKind::File),
        Some(FsEvent::Modified)
    );
    assert_eq!(
        kevent_to_fs_event(0, libc::NOTE_EXTEND, ResourceKind::Dir),
        Some(FsEvent::StructureChanged)
    );
}

#[test]
fn normalize_attrib_alone_is_metadata() {
    assert_eq!(
        kevent_to_fs_event(0, libc::NOTE_ATTRIB, ResourceKind::File),
        Some(FsEvent::MetadataChanged)
    );
}

#[test]
fn normalize_attrib_with_write_emits_write() {
    // WRITE > ATTRIB; the engine's debouncing handles the metadata
    // change as part of the same Settling burst.
    let fflags = libc::NOTE_ATTRIB | libc::NOTE_WRITE;
    assert_eq!(
        kevent_to_fs_event(0, fflags, ResourceKind::File),
        Some(FsEvent::Modified)
    );
}

#[test]
fn normalize_no_actionable_signal_returns_none() {
    assert_eq!(kevent_to_fs_event(0, 0, ResourceKind::File), None);
    assert_eq!(kevent_to_fs_event(0, 0, ResourceKind::Dir), None);
    assert_eq!(kevent_to_fs_event(0, 0, ResourceKind::Unknown), None);
}

#[test]
fn normalize_unknown_kind_defaults_to_modified() {
    assert_eq!(
        kevent_to_fs_event(0, libc::NOTE_WRITE, ResourceKind::Unknown),
        Some(FsEvent::Modified)
    );
}

// ── deadline math ──────────────────────────────────────────────────

#[test]
fn deadline_in_past_clamps_to_zero() {
    let past = Instant::now()
        .checked_sub(Duration::from_mins(1))
        .expect("60s before Instant::now() is representable");
    let ts = deadline_instant_to_timespec(past);
    assert_eq!(ts.tv_sec, 0);
    assert_eq!(ts.tv_nsec, 0);
}

#[test]
fn deadline_future_round_trip_within_a_second() {
    let dur = Duration::from_millis(500);
    let ts = deadline_instant_to_timespec(Instant::now() + dur);
    // The deadline is `now + 500ms` and `deadline_instant_to_timespec`
    // reads `Instant::now()` again internally, so the timespec is at
    // most 500ms and should be within ~50ms of that target.
    //
    // `tv_sec`/`tv_nsec` are signed (`i64`/`c_long`) on macOS/FreeBSD;
    // the conversion back to `u64`/`u32` is bounded by the sub-second
    // duration we just produced.
    let secs = u64::try_from(ts.tv_sec).expect("non-negative tv_sec");
    let nanos = u32::try_from(ts.tv_nsec).expect("non-negative, < 1s tv_nsec");
    let dur_ts = Duration::new(secs, nanos);
    assert!(dur_ts <= dur, "{dur_ts:?} <= {dur:?}");
    assert!(
        dur_ts > dur.saturating_sub(Duration::from_millis(50)),
        "{dur_ts:?} > {:?}",
        dur.saturating_sub(Duration::from_millis(50))
    );
}

// ── ffi sanity ─────────────────────────────────────────────────────

#[test]
fn duration_to_timespec_zero_yields_zero_components() {
    let ts = ffi::duration_to_timespec(Duration::ZERO);
    assert_eq!(ts.tv_sec, 0);
    assert_eq!(ts.tv_nsec, 0);
}

#[test]
fn duration_to_timespec_one_sec_one_nano() {
    let ts = ffi::duration_to_timespec(Duration::new(1, 1));
    assert_eq!(ts.tv_sec, 1);
    assert_eq!(ts.tv_nsec, 1);
}

#[test]
fn kevent_zeroed_is_default_state() {
    let ev = ffi::Kevent::zeroed();
    // `EVFILT_*` constants are negative on macOS / FreeBSD; zero is a
    // valid (and unused) bit pattern that we never treat as a real
    // filter, confirming the zero-init is "untriggered". `udata` of
    // zero round-trips to `None` (the wake-event sentinel).
    assert_eq!(ev.flags(), 0);
    assert_eq!(ev.fflags(), 0);
    assert!(ev.resource_id().is_none(), "zero udata decodes to None");
    // Zero `filter` is not `EVFILT_USER` (a negative value on both
    // BSDs), so an arbitrary user-ident probe rejects.
    assert!(
        !ev.is_user_event(0xDEAD_BEEF),
        "zero-init does not look like a user event"
    );
}
