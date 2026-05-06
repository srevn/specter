//! Re-registration on mask change — the per-FD `registered_fflags`
//! cache + `EV_ADD`-overwrites-fflags semantics from D11 / R2.
//!
//! These tests exercise [`KqueueWatcher::watch`]'s re-watch path: a
//! second `watch()` call on a resource that already holds an `OwnedFd`.
//! The watcher diffs the cached fflags against the translator's output
//! for the new `(opts.events, kind)` and re-registers via `EV_ADD` when
//! they differ. macOS / FreeBSD only.

#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, WatchOpts};
use specter_sensor::{FsWatcher, KqueueWatcher};
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Build a [`WatchOpts`] with the given event-class mask.
const fn opts(events: ClassSet) -> WatchOpts {
    WatchOpts { events }
}

/// Drain at least one event matching `pred` or hit `overall` deadline.
fn drain_until<F: Fn(&(ResourceId, FsEvent)) -> bool>(
    w: &mut KqueueWatcher,
    pred: F,
    overall: Duration,
) -> Vec<(ResourceId, FsEvent)> {
    let deadline = Instant::now() + overall;
    let mut out = Vec::new();
    while Instant::now() < deadline {
        let _ = w.poll_until(Some(Instant::now() + Duration::from_millis(50)), &mut out);
        if out.iter().any(&pred) {
            return out;
        }
    }
    out
}

fn drain_for(w: &mut KqueueWatcher, dur: Duration) -> Vec<(ResourceId, FsEvent)> {
    let mut out = Vec::new();
    let _ = w.poll_until(Some(Instant::now() + dur), &mut out);
    out
}

/// Re-watch with a widened mask should make new event classes deliverable.
/// Specifically: a CONTENT-only registration filters out
/// `MetadataChanged`; widening to `CONTENT | METADATA` (a fresh `Watch`
/// op the engine emits when `Resource.events_union` changes per D11)
/// must re-register the FD with `NOTE_ATTRIB`, and a subsequent chmod
/// then fires `MetadataChanged`.
#[test]
fn rewatch_with_widened_mask_delivers_new_classes() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "initial").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // First registration: CONTENT only. NOTE_ATTRIB is NOT installed, so
    // chmod must not fire MetadataChanged.
    w.watch(r, &path, opts(ClassSet::CONTENT)).unwrap();

    // Drain any pending registration acks / spurious events.
    let _ = drain_for(&mut w, Duration::from_millis(100));

    // chmod: should NOT fire MetadataChanged since NOTE_ATTRIB isn't
    // registered.
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(&path, perms).unwrap();
    let early = drain_for(&mut w, Duration::from_millis(300));
    assert!(
        !early
            .iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "CONTENT-only mask must not deliver MetadataChanged; got {early:?}"
    );

    // Re-register with widened mask. Same path, same resource — the
    // watcher takes the re-watch path, diffs cached fflags vs new, and
    // re-registers via EV_ADD with NOTE_ATTRIB now in the mask.
    w.watch(r, &path, opts(ClassSet::CONTENT | ClassSet::METADATA))
        .unwrap();

    // Now chmod fires MetadataChanged.
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o644);
    std::fs::set_permissions(&path, perms).unwrap();
    let post = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::MetadataChanged,
        Duration::from_secs(2),
    );
    assert!(
        post.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "after re-register with METADATA, chmod should fire MetadataChanged; got {post:?}"
    );

    drop(w);
}

/// Re-watch with a narrowed mask should drop classes from the
/// registration. Specifically: an initial `STRUCTURE | METADATA`
/// registration on a Dir delivers both `StructureChanged` on child
/// write and `MetadataChanged` on chmod. Re-registering with
/// `STRUCTURE` only must remove `NOTE_ATTRIB`, so chmod no longer
/// fires.
#[test]
fn rewatch_with_narrowed_mask_drops_classes() {
    let tmp = TempDir::new().unwrap();
    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // First: STRUCTURE | METADATA on the dir.
    w.watch(
        r,
        tmp.path(),
        opts(ClassSet::STRUCTURE | ClassSet::METADATA),
    )
    .unwrap();
    let _ = drain_for(&mut w, Duration::from_millis(100));

    // Verify METADATA is currently registered: chmod fires.
    let mut perms = std::fs::metadata(tmp.path()).unwrap().permissions();
    perms.set_mode(0o700);
    std::fs::set_permissions(tmp.path(), perms).unwrap();
    let pre = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::MetadataChanged,
        Duration::from_secs(2),
    );
    assert!(
        pre.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "pre-narrow: chmod should fire MetadataChanged; got {pre:?}"
    );

    // Narrow to STRUCTURE only. Re-watch path: re-registers without
    // NOTE_ATTRIB.
    w.watch(r, tmp.path(), opts(ClassSet::STRUCTURE)).unwrap();
    let _ = drain_for(&mut w, Duration::from_millis(100));

    // chmod again — must NOT fire MetadataChanged anymore.
    let mut perms = std::fs::metadata(tmp.path()).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(tmp.path(), perms).unwrap();
    let post = drain_for(&mut w, Duration::from_millis(300));
    assert!(
        !post
            .iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "post-narrow: METADATA dropped, chmod must not fire MetadataChanged; got {post:?}"
    );

    // STRUCTURE still works — child write fires StructureChanged.
    std::fs::write(tmp.path().join("child.txt"), "x").unwrap();
    let post_struct = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::StructureChanged,
        Duration::from_secs(2),
    );
    assert!(
        post_struct
            .iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::StructureChanged),
        "post-narrow: STRUCTURE still active, child write should fire StructureChanged; got {post_struct:?}"
    );

    drop(w);
}

/// Re-watch with the same mask is a no-op — the watcher's cache catches
/// it and skips the syscall. Observationally: the fd's behavior is
/// unchanged. We exercise this by checking that a normal write still
/// fires `Modified` after a same-mask re-watch (i.e., re-watching
/// didn't accidentally clear the fflags).
#[test]
fn rewatch_with_same_mask_preserves_registration() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    let opts = opts(ClassSet::CONTENT);
    w.watch(r, &path, opts).unwrap();
    // Same opts twice: hits the cache-diff `noop` branch.
    w.watch(r, &path, opts).unwrap();

    std::fs::write(&path, "y").unwrap();
    let out = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::Modified,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::Modified),
        "post no-op rewatch, write should still fire Modified; got {out:?}"
    );

    drop(w);
}

/// Per design §10.5 — suppression and mask changes interact. A re-watch
/// after `suppress()` must keep the resource silenced even when the new
/// mask widens. The watcher composes `EV_ADD | EV_CLEAR | EV_DISABLE` on
/// a single change record so the kernel-side filter never observes an
/// enabled state mid-update.
#[test]
fn rewatch_preserves_suppress_state() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    w.watch(r, &path, opts(ClassSet::CONTENT)).unwrap();
    w.suppress(r);
    let _ = drain_for(&mut w, Duration::from_millis(100));

    // Re-watch with widened mask. Per §10.5, suppression is preserved.
    w.watch(r, &path, opts(ClassSet::CONTENT | ClassSet::METADATA))
        .unwrap();

    // Even though the new mask covers METADATA, chmod must not deliver
    // — delivery is still suppressed.
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(&path, perms).unwrap();
    let suppressed = drain_for(&mut w, Duration::from_millis(300));
    assert!(
        !suppressed.iter().any(|(rid, _)| *rid == r),
        "suppress should silence delivery across re-register; got {suppressed:?}"
    );

    // Restore delivery and confirm the new METADATA mask actually took
    // effect — chmod after unsuppress fires MetadataChanged.
    w.unsuppress(r);
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o644);
    std::fs::set_permissions(&path, perms).unwrap();
    let restored = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::MetadataChanged,
        Duration::from_secs(2),
    );
    assert!(
        restored
            .iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "after unsuppress, the widened mask should deliver MetadataChanged; got {restored:?}"
    );

    drop(w);
}

/// F-HIGH-1 race property: events queued in the kernel filter BEFORE the
/// initial `suppress()` must remain silenced across a subsequent
/// `watch()` (re-register) call on the suppressed FD. The re-register's
/// single-syscall `EV_ADD | EV_CLEAR | EV_DISABLE` change record collapses
/// the prior two-syscall window — under the old shape an `EV_ADD`-then-
/// `EV_DISABLE` sequence transiently enabled the kernel-side filter, so
/// any pending event on it would become deliverable on a concurrent
/// `kevent` drain.
///
/// The watcher's `poll_until` is the only consumer of the kqueue fd, so
/// the race is single-thread quiescent — but the property the kernel
/// guarantees is "no transient enable", and the test pins that contract
/// regardless of when delivery fires. The harness writes BEFORE
/// `suppress`, then issues a re-watch with a widened mask, then drains.
/// Pre-fix, the drain after re-watch would surface the queued
/// `Modified`; post-fix it stays buffered behind the disable bit until
/// `unsuppress`.
#[test]
fn rewatch_on_suppressed_fd_does_not_leak_pending_events() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // Watch before any modification so the kernel filter is live.
    w.watch(r, &path, opts(ClassSet::CONTENT)).unwrap();

    // Write — queues a NOTE_WRITE in the kernel filter. Drain it once so
    // we know the event landed; this also clears any registration-time
    // ack bits.
    std::fs::write(&path, "y").unwrap();
    let initial = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::Modified,
        Duration::from_secs(2),
    );
    assert!(
        initial
            .iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::Modified),
        "initial write must deliver before suppress; got {initial:?}",
    );

    // Suppress, then queue another write while suppressed. Under
    // EV_DISABLE the filter still buffers events — they accumulate
    // behind the disable bit and stay invisible to drains.
    w.suppress(r);
    std::fs::write(&path, "z").unwrap();
    let suppressed_drain = drain_for(&mut w, Duration::from_millis(200));
    assert!(
        !suppressed_drain.iter().any(|(rid, _)| *rid == r),
        "writes during suppress must not deliver; got {suppressed_drain:?}",
    );

    // Re-watch with a widened mask. The single-syscall change record
    // re-registers AND keeps the disable bit set; the buffered NOTE_WRITE
    // does not slip out.
    w.watch(r, &path, opts(ClassSet::CONTENT | ClassSet::METADATA))
        .unwrap();
    let post_rewatch = drain_for(&mut w, Duration::from_millis(300));
    assert!(
        !post_rewatch.iter().any(|(rid, _)| *rid == r),
        "re-watch on a suppressed FD must keep buffered events silenced; \
         got {post_rewatch:?} — F-HIGH-1 regression",
    );

    // Sanity check: unsuppress reveals the buffered event(s), confirming
    // they were merely silenced (not lost) and the new mask took effect.
    w.unsuppress(r);
    let restored = drain_until(
        &mut w,
        |(rid, _)| *rid == r,
        Duration::from_secs(2),
    );
    assert!(
        restored.iter().any(|(rid, _)| *rid == r),
        "unsuppress must flush buffered events; got {restored:?}",
    );

    drop(w);
}

/// `unwatch` clears the per-FD fflags cache so a subsequent fresh
/// `watch` opens a new FD. This is the cache-lifecycle invariant.
#[test]
fn unwatch_then_watch_starts_fresh() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    w.watch(r, &path, opts(ClassSet::CONTENT)).unwrap();
    w.unwatch(r);
    // Fresh watch (FD reopened, cache repopulated). Observable check:
    // a subsequent write fires Modified normally.
    w.watch(r, &path, opts(ClassSet::CONTENT)).unwrap();

    std::fs::write(&path, "y").unwrap();
    let out = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::Modified,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::Modified),
        "post unwatch+watch, write should fire Modified; got {out:?}"
    );

    drop(w);
}
