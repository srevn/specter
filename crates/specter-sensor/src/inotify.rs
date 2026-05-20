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
//! 32-bit Linux is gated out at compile time. The `wd → ResourceId`
//! mapping does not require pointer-width parity (the watcher pays a
//! `BTreeMap` cell), but the kqueue sibling does, and we keep the rule
//! uniform — every backend assumes a 64-bit address space, every test
//! fixture is 64-bit.

#[cfg(target_pointer_width = "32")]
compile_error!(
    "specter-sensor: 32-bit Linux is unsupported in v1 — keeping the \
     pointer-width rule uniform with the kqueue branch."
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
