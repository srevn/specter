//! The engine's empty-path sentinel for the probe / watch wire.
//!
//! [`Tree::path_of`](specter_core::Tree::path_of) is honestly `Option`: a stale `ResourceId` has no
//! `Resource`, hence no path. The probe / watch wire, however, encodes "no path" as an *empty* path
//! — the walker treats an empty `target_path` as `ProbeOutcome::Vanished`. This module owns that
//! one translation so the empty-as-`Vanished` protocol lives at a single engine-boundary site,
//! never smuggled into `core::Tree` (which stays an honest `Option`).

use std::path::Path;
use std::sync::{Arc, LazyLock};

/// The shared empty `Arc<Path>` a stale-id `path_of` maps to (`None` ⇒ the walker observes
/// `Vanished`). One process-wide allocation; every cold stale-path site `Arc::clone`s it rather
/// than allocating per call. Use as `tree.path_of(id).unwrap_or_else(empty_path)`.
pub(crate) fn empty_path() -> Arc<Path> {
    static EMPTY: LazyLock<Arc<Path>> = LazyLock::new(|| Arc::from(Path::new("")));
    EMPTY.clone()
}
