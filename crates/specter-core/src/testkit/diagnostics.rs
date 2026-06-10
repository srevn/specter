//! Test ergonomics for reading minted ids out of a [`StepOutput`].
//!
//! [`crate::Input::AttachSub`] surfaces its minted [`SubId`] through the
//! [`crate::Diagnostic::SubAttached`] stream in `StepOutput.diagnostics`. The bin's
//! `reconcile_loader_from_diagnostics` walks the full stream because hot reload can attach many
//! subs in one [`crate::Input::ConfigDiff`] step; tests typically attach exactly one at a time, so
//! the "give me the first id" pattern is the natural test-only ergonomic.
//!
//! Returning `None` keeps the path-rejection contract observable in tests: an
//! [`crate::Diagnostic::AttachPathInvalid`] outcome doesn't produce a `SubAttached`, so this helper
//! returns `None` and the test body can route to the negative-case assertion (or `.expect(...)`
//! when the test pins the positive case).

use crate::diag::Diagnostic;
use crate::ids::SubId;
use crate::output::StepOutput;

/// First [`SubId`] minted by a [`crate::Diagnostic::SubAttached`].
///
/// Returns `None` if no such diagnostic is present — i.e., the attach was rejected at the path gate
/// ([`crate::Diagnostic::AttachPathInvalid`]) or the output is from a non-attach step.
#[must_use]
pub fn first_attached_sub(out: &StepOutput) -> Option<SubId> {
    out.diagnostics.iter().find_map(|d| match d {
        Diagnostic::SubAttached { sub, .. } => Some(*sub),
        _ => None,
    })
}
