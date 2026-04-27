//! `StepOutput`. Field order and `TinyVec` inline capacities mirror the spec;
//! sort guarantees are enforced by the engine (callers see already-sorted
//! slices).

use crate::diag::Diagnostic;
use crate::effect::Effect;
use crate::op::{ProbeOp, WatchOp};
use tinyvec::TinyVec;

#[derive(Debug, Default, Clone)]
pub struct StepOutput {
    pub watch_ops: TinyVec<[WatchOp; 2]>,
    pub probe_ops: TinyVec<[ProbeOp; 4]>,
    pub effects: TinyVec<[Effect; 2]>,
    pub diagnostics: TinyVec<[Diagnostic; 2]>,
}
