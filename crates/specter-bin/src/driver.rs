//! Engine driver — the bin's main-thread loop, split across three
//! focused submodules with the spine here.
//!
//! [`EngineDriver`] owns the [`Engine`], the [`Loader`], the
//! engine-side channel bundle, the prober [`Arc`] clone and a
//! wake-handle clone. This module holds the struct and its lifecycle
//! ([`EngineDriver::new`], [`EngineDriver::run_initial_attach`],
//! [`EngineDriver::run`]) plus the cancel-first shutdown drain
//! (`begin_shutdown`). The load-bearing work is next to it:
//!
//! - [`tick`] — one pass of the drain order (sensor → timers → reload
//!   → config-settle → effects → block). The hot loop; new
//!   inbound-path work lands there.
//! - [`reload`] — the SIGHUP + auto-reload settle pipeline.
//! - [`forward`] — ships a `StepOutput` downstream and maps a
//!   `Diagnostic` to tracing.
//!
//! `run_initial_attach` walks `loader.current_config` in source order,
//! attaching each Sub / Promoter and forwarding the resulting output
//! immediately so the watcher / prober see work as it lands. `run`
//! wraps [`EngineDriver::tick`] until shutdown. All file I/O is on
//! this thread — no Mutex.

mod forward;
mod reload;
mod tick;

use crate::app::CliLogOverrides;
use crate::channels::EngineSide;
use crate::loader::Loader;
use crate::observability::ObservabilityHandle;
use specter_core::Input;
use specter_engine::Engine;
use specter_sensor::{Prober, WakeHandle};
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

/// Reason the driver loop exited. Returned from [`EngineDriver::run`].
///
/// v1 has only the `Shutdown` variant — every path that could exit the
/// loop without a shutdown signal (sensor channel disconnect) currently
/// also routes through `TickOutcome::Shutdown` per [`EngineDriver::tick`].
/// The enum exists so v2 (recovery / restart) has a structural seam
/// without breaking the [`EngineDriver::run`] return type.
#[derive(Debug, Eq, PartialEq)]
pub enum ExitReason {
    /// `shutdown_engine_rx` fired (operator-driven, normal path), OR
    /// every input channel disconnected (upstream thread crash; v1
    /// treats both as terminal-graceful).
    Shutdown,
}

/// Outcome of a single [`EngineDriver::tick`] call. The loop wrapper
/// matches on this; explicit enum is friendlier than a bool.
#[derive(Debug, Eq, PartialEq)]
pub enum TickOutcome {
    /// Inputs drained; loop again.
    Continue,
    /// Operator signal or sensor disconnect. The tick has already run
    /// the cancel-first probe drain ([`EngineDriver::begin_shutdown`]),
    /// so the engine holds no armed probe: tearing the driver down
    /// (the bin's `drop(driver)`) will not trip the linear `ProbeSlot`
    /// Drop guard.
    Shutdown,
}

/// Engine driver — see module rustdoc.
pub struct EngineDriver {
    engine: Engine,
    loader: Loader,
    config_path: PathBuf,
    /// CLI overrides applied to `[log]` at startup. Re-applied on every
    /// SIGHUP-driven reload so CLI precedence stays consistent across
    /// the process lifetime (`CLI > config > default`).
    cli_log_overrides: CliLogOverrides,
    /// Subscriber handle for runtime updates (`set_level`,
    /// `reopen_file`). Held here so `handle_reload` can fire both on
    /// SIGHUP without going through the loader.
    obs_handle: ObservabilityHandle,
    sides: EngineSide,
    prober: Arc<dyn Prober>,
    wake_handle: Box<dyn WakeHandle>,
    /// Auto-reload settle deadline — armed by the config-event drain
    /// (the watcher thread's `try_send` or a test rig's manual
    /// `try_send`), expires after [`CONFIG_SETTLE`] of quiet, at
    /// which point the driver runs the
    /// lstat-vs-`loader.config_meta` filter and (on drift) calls
    /// [`Self::handle_reload`]. Reset to `None` on expiry and
    /// re-armed per pulse (settle resets, so sustained bursts defer
    /// the reload until the edits actually settle).
    ///
    /// Two consumers:
    /// - The [`Self::tick`] timeout math feeds the deadline into
    ///   `Select::ready_timeout` so the driver wakes precisely when
    ///   the window expires, not on the next sensor / effect pulse.
    /// - [`Self::apply_config_settle_expiry`] gates the lstat call
    ///   on `now >= deadline` so the engine thread never lstats
    ///   before the settle window has elapsed.
    config_settle_until: Option<Instant>,
}

impl std::fmt::Debug for EngineDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineDriver")
            .field("loader", &self.loader)
            .field("config_path", &self.config_path)
            .field("cli_log_overrides", &self.cli_log_overrides)
            .field("obs_handle", &self.obs_handle)
            .finish_non_exhaustive()
    }
}

impl EngineDriver {
    #[must_use]
    pub fn new(
        engine: Engine,
        loader: Loader,
        config_path: PathBuf,
        cli_log_overrides: CliLogOverrides,
        obs_handle: ObservabilityHandle,
        sides: EngineSide,
        prober: Arc<dyn Prober>,
        wake_handle: Box<dyn WakeHandle>,
    ) -> Self {
        Self {
            engine,
            loader,
            config_path,
            cli_log_overrides,
            obs_handle,
            sides,
            prober,
            wake_handle,
            config_settle_until: None,
        }
    }

    /// Attach every active Sub and Promoter from
    /// `loader.current_config` in source order. Disabled entries are
    /// filtered out via [`Config::active_watches`] /
    /// [`Config::active_promoters`] — they remain in the raw `Vec`s
    /// for introspection but never reach the engine, mirroring the
    /// "disabled = absent" discipline the diff layer applies to
    /// hot-reload.
    ///
    /// Each [`StepOutput`] is forwarded as we go so the watcher /
    /// prober receive ops as the engine emits them — a single
    /// startup-sized `ConfigDiff` would batch the entire attach into
    /// one output and stall the watcher behind the post-call
    /// `forward`. Hot-reload (in `reload.rs`) deliberately uses the
    /// inverse pattern — a single batched `Input::ConfigDiff` — because
    /// reload diffs are typically small. Revisit if those diffs grow
    /// large enough to stall the watcher behind a single `forward`.
    ///
    /// No bin-side reconciliation: the engine owns `name → id`
    /// resolution through its registries' `by_name` indices. The
    /// `SubAttached` / `PromoterAttached` diagnostics are pure operator
    /// narration, logged via `forward`.
    ///
    /// Returns [`ControlFlow::Break`] if any `forward` observed
    /// shutdown (operator pulse mid-attach or `watch_ops_tx`
    /// disconnect). On `Break` we run [`Self::begin_shutdown`] before
    /// returning — an attached Sub leaves the Profile in a
    /// Seed-Verifying state with an armed `ProbeSlot`, and a caller
    /// that just drops the driver would trip
    /// `ProbeSlot::drop`'s linear-edge tripwire. Containing the
    /// probe drain inside `run_initial_attach` keeps the lifecycle
    /// discipline encapsulated; the caller (`app.rs`) stays a thin
    /// branch on the `ControlFlow` return.
    pub(crate) fn run_initial_attach(&mut self) -> ControlFlow<()> {
        let now = Instant::now();
        // Snapshot the active spec lists: `self.engine.step` needs
        // `&mut self`, so the `&self` borrow on `loader.current_config`
        // cannot be held across the loop.
        let watch_specs: Vec<_> = self
            .loader
            .current_config
            .active_watches()
            .cloned()
            .collect();
        let promoter_specs: Vec<_> = self
            .loader
            .current_config
            .active_promoters()
            .cloned()
            .collect();
        for spec in watch_specs {
            let req = spec.to_attach_request();
            let out = self.engine.step(Input::AttachSub(req), now);
            if self.forward(out).is_break() {
                let _ = self.begin_shutdown();
                return ControlFlow::Break(());
            }
        }
        for spec in promoter_specs {
            let req = spec.to_attach_request();
            let out = self.engine.step(Input::AttachPromoter(req), now);
            if self.forward(out).is_break() {
                let _ = self.begin_shutdown();
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }

    /// Loop wrapping [`Self::tick`] until shutdown.
    ///
    /// MUST NOT be wrapped in `catch_unwind`: `ProbeSlot`'s in-unwind
    /// silence (`specter_core::probe`) depends on a mid-`step` panic
    /// being fatal — catching it here would let the daemon carry on
    /// with a probe-bearing state torn down mid-flight.
    pub fn run(&mut self) -> ExitReason {
        loop {
            match self.tick() {
                TickOutcome::Continue => {}
                TickOutcome::Shutdown => return ExitReason::Shutdown,
            }
        }
    }

    /// Cancel-first shutdown teardown, run once when [`Self::tick`]
    /// resolves to shutdown (operator signal or sensor disconnect).
    ///
    /// The linear `ProbeSlot` Drop tripwire panics if the `Engine` is
    /// dropped (the bin's `drop(driver)`) with a probe still armed,
    /// and a graceful shutdown routinely coincides with one in flight
    /// (settle / verify / rebase / descent). Disarm every owner's slot
    /// and forward the resulting `Cancel`s to the prober — the same
    /// disarm-then-`Cancel` discipline the engine applies to its
    /// internal abandon sites, now at the process boundary. After this
    /// returns the engine holds no armed probe, so dropping it is
    /// silent and [`TickOutcome::Shutdown`] means "drained, safe to
    /// tear down".
    #[must_use]
    fn begin_shutdown(&mut self) -> TickOutcome {
        let out = self.engine.cancel_all_in_flight_probes();
        // INVARIANT: cancel_all_in_flight_probes emits exclusively
        // `ProbeOp::Cancel` ops (see `engine::probe::cancel_owner_probe`
        // — the disarm-then-`Cancel` choke this drain iterates over).
        // `watch_ops` and `effects` are therefore structurally empty,
        // so `forward`'s outbound `crossbeam::select!` arms never
        // execute on this `StepOutput`; the `ControlFlow` return is
        // structurally `Continue`. The cancels dispatch through
        // `forward`'s probe arm directly to the prober (no channel,
        // no shutdown race), so the discard is intentional. A future
        // refactor adding non-probe ops to
        // `cancel_all_in_flight_probes` must thread `Break` here.
        debug_assert!(
            out.watch_ops.is_empty() && out.effects().is_empty(),
            "cancel_all_in_flight_probes must emit only ProbeOp::Cancel",
        );
        let _ = self.forward(out);
        TickOutcome::Shutdown
    }
}

#[cfg(test)]
mod tests;
