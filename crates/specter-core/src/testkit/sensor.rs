//! Sensor-shaped test harness.
//!
//! **Not** an `FsWatcher` impl — that trait lives in `specter-sensor`,
//! which `specter-core` may not depend on. Instead, `MockSensor`
//! is a recorder + injector over the `StepOutput`/`Input` boundary.
//!
//! `observe` aggregates every `StepOutput` field; the asserter helpers
//! (`is_watched`, `is_suppressed`, …) walk the captured op stream and
//! report the *net* state for a Resource — `Watch + Unwatch` cancels out,
//! so the helpers reflect what the Sensor would have observed at the
//! moment of the last edge.

use crate::{
    Diagnostic, Effect, FsEvent, Input, ProbeOp, ProbeResponse, ResourceId, StepOutput,
    WatchFailure, WatchOp,
};

#[derive(Debug, Default, Clone)]
pub struct MockSensor {
    pub watch_ops: Vec<WatchOp>,
    pub probe_ops: Vec<ProbeOp>,
    pub effects: Vec<Effect>,
    pub diagnostics: Vec<Diagnostic>,
}

impl MockSensor {
    /// Drain a `StepOutput`'s op streams into the recorder. The caller
    /// still owns the `StepOutput` — the recorder snapshots references.
    pub fn observe(&mut self, out: &StepOutput) {
        self.watch_ops.extend(out.watch_ops.iter().cloned());
        self.probe_ops.extend(out.probe_ops.iter().cloned());
        self.effects.extend(out.effects.iter().cloned());
        self.diagnostics.extend(out.diagnostics.iter().cloned());
    }

    pub fn reset(&mut self) {
        self.watch_ops.clear();
        self.probe_ops.clear();
        self.effects.clear();
        self.diagnostics.clear();
    }

    /// First-emission predicate: `true` iff a `WatchOp::Watch` for `r` was
    /// observed at any point. Does **not** net out a subsequent
    /// `WatchOp::Unwatch` — prefer [`is_watched`](Self::is_watched).
    #[deprecated(note = "use `is_watched`; this only checks first-emission, hides Unwatch")]
    #[must_use]
    pub fn watched(&self, r: ResourceId) -> bool {
        self.watch_ops.iter().any(|op| {
            matches!(op,
            WatchOp::Watch { resource, .. } if *resource == r)
        })
    }

    /// Net-balance predicate: `true` iff the **last** `Watch`/`Unwatch` op
    /// observed for `r` was `Watch`. Returns `false` if no op for `r` has
    /// ever been observed.
    #[must_use]
    pub fn is_watched(&self, r: ResourceId) -> bool {
        matches!(self.last_watch_edge(r), Some(Edge::Set))
    }

    /// Complement of [`is_watched`](Self::is_watched). Includes the
    /// "never observed" case — `r` not appearing in the captured stream
    /// reports as unwatched.
    #[must_use]
    pub fn is_unwatched(&self, r: ResourceId) -> bool {
        !self.is_watched(r)
    }

    /// Net-balance predicate for `Suppress`/`Unsuppress` ops: `true` iff
    /// the **last** suppress-edge op observed for `r` was `Suppress`.
    #[must_use]
    pub fn is_suppressed(&self, r: ResourceId) -> bool {
        matches!(self.last_suppress_edge(r), Some(Edge::Set))
    }

    /// Complement of [`is_suppressed`](Self::is_suppressed); also covers
    /// "never observed".
    #[must_use]
    pub fn is_unsuppressed(&self, r: ResourceId) -> bool {
        !self.is_suppressed(r)
    }

    fn last_watch_edge(&self, r: ResourceId) -> Option<Edge> {
        self.watch_ops.iter().rev().find_map(|op| match op {
            WatchOp::Watch { resource, .. } if *resource == r => Some(Edge::Set),
            WatchOp::Unwatch { resource } if *resource == r => Some(Edge::Cleared),
            _ => None,
        })
    }

    fn last_suppress_edge(&self, r: ResourceId) -> Option<Edge> {
        self.watch_ops.iter().rev().find_map(|op| match op {
            WatchOp::Suppress { resource } if *resource == r => Some(Edge::Set),
            WatchOp::Unsuppress { resource } if *resource == r => Some(Edge::Cleared),
            _ => None,
        })
    }

    // Typed `Input` constructors. Test code spells injections as
    // `MockSensor::fs_event(rid, FsEvent::Modified)` without having to
    // remember the variant payload shape.

    #[must_use]
    pub const fn fs_event(resource: ResourceId, event: FsEvent) -> Input {
        Input::FsEvent { resource, event }
    }

    #[must_use]
    pub const fn probe_response(response: ProbeResponse) -> Input {
        Input::ProbeResponse(response)
    }

    /// Mirror of [`fs_event`](Self::fs_event) for FD-pressure recovery —
    /// wraps the rejected op + typed [`WatchFailure`] into the engine
    /// input shape.
    #[must_use]
    pub const fn watch_op_rejected(
        resource: ResourceId,
        op: WatchOp,
        failure: WatchFailure,
    ) -> Input {
        Input::WatchOpRejected {
            resource,
            op,
            failure,
        }
    }
}

#[derive(Copy, Clone, Eq, PartialEq)]
enum Edge {
    Set,
    Cleared,
}

#[cfg(test)]
mod tests {
    use super::MockSensor;
    use crate::{
        ClassSet, Diagnostic, Input, ProbeOp, ResourceId, ResourceKind, StepOutput, WatchOp,
    };
    use slotmap::SlotMap;
    use std::path::PathBuf;

    fn fresh_resource_ids(n: usize) -> Vec<ResourceId> {
        let mut sm = SlotMap::<ResourceId, ()>::with_key();
        (0..n).map(|_| sm.insert(())).collect()
    }

    #[test]
    fn observe_aggregates_across_steps() {
        let ids = fresh_resource_ids(2);
        let mut sensor = MockSensor::default();

        let mut step_one = StepOutput::default();
        step_one.watch_ops.push(WatchOp::Watch {
            resource: ids[0],
            path: PathBuf::from("/tmp/a"),
            kind: ResourceKind::Unknown,
            events: ClassSet::EMPTY,
        });
        step_one.probe_ops.push(ProbeOp::Cancel {
            profile: crate::ProfileId::default(),
        });

        let mut step_two = StepOutput::default();
        step_two
            .watch_ops
            .push(WatchOp::Unwatch { resource: ids[1] });

        sensor.observe(&step_one);
        sensor.observe(&step_two);

        assert_eq!(sensor.watch_ops.len(), 2);
        assert_eq!(sensor.probe_ops.len(), 1);

        sensor.reset();
        assert!(sensor.watch_ops.is_empty());
        assert!(sensor.probe_ops.is_empty());
        assert!(sensor.effects.is_empty());
        assert!(sensor.diagnostics.is_empty());
    }

    #[test]
    fn observe_captures_effects_and_diagnostics() {
        let ids = fresh_resource_ids(1);

        let mut out = StepOutput::default();
        out.diagnostics
            .push(Diagnostic::EventOnUnwatchedResource { resource: ids[0] });

        let mut sensor = MockSensor::default();
        sensor.observe(&out);

        assert_eq!(sensor.diagnostics.len(), 1);
        // Effects are populated by the engine in P5+; an empty `StepOutput`
        // still flows through `observe` cleanly.
        assert!(sensor.effects.is_empty());
    }

    #[test]
    #[allow(deprecated)]
    fn watched_discriminates_by_resource_id() {
        let ids = fresh_resource_ids(2);
        let (watched, unwatched) = (ids[0], ids[1]);

        let mut sensor = MockSensor::default();
        let mut out = StepOutput::default();
        out.watch_ops.push(WatchOp::Watch {
            resource: watched,
            path: PathBuf::from("/tmp/x"),
            kind: ResourceKind::Unknown,
            events: ClassSet::EMPTY,
        });
        sensor.observe(&out);

        assert!(sensor.watched(watched));
        assert!(!sensor.watched(unwatched));

        // `Unwatch` is not a `Watch` — must still discriminate.
        let mut other = StepOutput::default();
        other.watch_ops.push(WatchOp::Unwatch {
            resource: unwatched,
        });
        sensor.observe(&other);
        assert!(!sensor.watched(unwatched));
    }

    #[test]
    fn is_watched_after_watch() {
        let ids = fresh_resource_ids(1);
        let r = ids[0];

        let mut out = StepOutput::default();
        out.watch_ops.push(WatchOp::Watch {
            resource: r,
            path: PathBuf::from("/tmp/z"),
            kind: ResourceKind::Unknown,
            events: ClassSet::EMPTY,
        });

        let mut sensor = MockSensor::default();
        sensor.observe(&out);

        assert!(sensor.is_watched(r));
        assert!(!sensor.is_unwatched(r));
    }

    #[test]
    fn is_watched_false_after_watch_then_unwatch() {
        let ids = fresh_resource_ids(1);
        let r = ids[0];

        let mut out = StepOutput::default();
        out.watch_ops.push(WatchOp::Watch {
            resource: r,
            path: PathBuf::from("/tmp/w"),
            kind: ResourceKind::Unknown,
            events: ClassSet::EMPTY,
        });
        out.watch_ops.push(WatchOp::Unwatch { resource: r });

        let mut sensor = MockSensor::default();
        sensor.observe(&out);

        assert!(!sensor.is_watched(r));
        assert!(sensor.is_unwatched(r));
    }

    #[test]
    fn is_watched_after_watch_unwatch_watch() {
        let ids = fresh_resource_ids(1);
        let r = ids[0];

        let mut out = StepOutput::default();
        out.watch_ops.push(WatchOp::Watch {
            resource: r,
            path: PathBuf::from("/tmp/w"),
            kind: ResourceKind::Unknown,
            events: ClassSet::EMPTY,
        });
        out.watch_ops.push(WatchOp::Unwatch { resource: r });
        out.watch_ops.push(WatchOp::Watch {
            resource: r,
            path: PathBuf::from("/tmp/w"),
            kind: ResourceKind::Unknown,
            events: ClassSet::EMPTY,
        });

        let mut sensor = MockSensor::default();
        sensor.observe(&out);

        // Net-balance ⇒ Watched (the last edge for `r` is `Set`).
        assert!(sensor.is_watched(r));
    }

    #[test]
    fn is_watched_false_for_never_observed_resource() {
        let ids = fresh_resource_ids(1);
        let sensor = MockSensor::default();

        assert!(!sensor.is_watched(ids[0]));
        assert!(sensor.is_unwatched(ids[0]));
    }

    #[test]
    fn is_suppressed_after_suppress() {
        let ids = fresh_resource_ids(1);
        let r = ids[0];

        let mut out = StepOutput::default();
        out.watch_ops.push(WatchOp::Suppress { resource: r });

        let mut sensor = MockSensor::default();
        sensor.observe(&out);

        assert!(sensor.is_suppressed(r));
        assert!(!sensor.is_unsuppressed(r));
    }

    #[test]
    fn is_suppressed_false_after_suppress_then_unsuppress() {
        let ids = fresh_resource_ids(1);
        let r = ids[0];

        let mut out = StepOutput::default();
        out.watch_ops.push(WatchOp::Suppress { resource: r });
        out.watch_ops.push(WatchOp::Unsuppress { resource: r });

        let mut sensor = MockSensor::default();
        sensor.observe(&out);

        assert!(!sensor.is_suppressed(r));
        assert!(sensor.is_unsuppressed(r));
    }

    #[test]
    fn watch_op_rejected_constructor_yields_input() {
        // `EMFILE` (Too many open files) is 24 on macOS / FreeBSD / Linux —
        // hard-coded here so `core` keeps its no-`libc` discipline.
        const EMFILE: i32 = 24;

        let ids = fresh_resource_ids(1);
        let r = ids[0];
        let op = WatchOp::Watch {
            resource: r,
            path: PathBuf::from("/tmp/p"),
            kind: ResourceKind::Unknown,
            events: ClassSet::EMPTY,
        };
        let input = MockSensor::watch_op_rejected(
            r,
            op,
            crate::WatchFailure::Pressure { errno: EMFILE },
        );
        match input {
            Input::WatchOpRejected {
                resource,
                op: WatchOp::Watch { .. },
                failure,
            } => {
                assert_eq!(resource, r);
                assert_eq!(failure, crate::WatchFailure::Pressure { errno: EMFILE });
            }
            _ => panic!("expected WatchOpRejected{{ Watch, Pressure(EMFILE) }}"),
        }
    }
}
