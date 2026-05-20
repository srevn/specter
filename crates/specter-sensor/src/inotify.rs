//! inotify-backed `FsWatcher` — Linux only.
//!
//! Layered (mirror of [`crate::kqueue`]):
//! - [`ffi`]: thin libc wrappers (the only `unsafe` surface).
//! - [`record`]: parser for the `inotify_event` record format.
//! - [`translate`]: `(ClassSet, ResourceKind) → mask`.
//! - [`normalize`]: `(mask, kind) → FsEvent`.
//! - [`wake`]: cross-thread interruption via eventfd.
//! - [`watcher`]: state-bearing `FsWatcher` impl.
//!
//! 32-bit Linux is gated out at compile time. The inotify watcher
//! itself does not require pointer-width parity — `wd → ResourceId`
//! routing pays a `BTreeMap` cell rather than stuffing the id into a
//! kernel-supplied slot. The gate exists to keep the rule uniform
//! with the kqueue sibling (which packs `ResourceId` into
//! `kevent.udata`, a `*mut c_void`) so the test fixtures stay
//! 64-bit-only across both backends.

#[cfg(target_pointer_width = "32")]
compile_error!(
    "specter-sensor: 32-bit Linux is unsupported in v1 — kept uniform \
     with the kqueue branch; see the module docs."
);

mod config_watch;
mod ffi;
mod normalize;
mod record;
mod translate;
mod wake;
mod watcher;

pub use config_watch::InotifyConfigWatcher;
pub use watcher::InotifyWatcher;
