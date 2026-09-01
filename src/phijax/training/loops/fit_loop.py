from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import jax
import numpy as np

from phijax.balancers import BalancerUpdate
from phijax.callbacks import TrainerContext
from phijax.core import BasePhiModule, PhiModuleContext
from phijax.metrics import _collect_module_metrics, _LoggedMetric, _metric_is_scalar
from phijax.training.lifecycle import TaskLifecycle
from phijax.training.loggers import scalar_metrics
from phijax.training.results import FitResult
from phijax.training.state import TrainState
from phijax.training.steps import TrainStep

if TYPE_CHECKING:
    from phijax.training.trainer import BatchSource, Trainer

log = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _BalancerUpdateRuntime:
    """Store the resolved host schedule for one adaptive balancer update.

    Attributes:
        update: Independently compiled functional balancer update.
        batches: Fixed diagnostic batches, or `None` to use the current training batch.
        every_n_steps: Positive interval between updates.
        update_start_step: Nonnegative absolute step anchoring the update cadence.
    """

    update: BalancerUpdate
    batches: Any | None
    every_n_steps: int
    update_start_step: int


class _FitLoop:
    """Run host-scheduled training around one compiled numerical update."""

    def __init__(self, trainer: Trainer) -> None:
        """Bind the loop to its owning Trainer.

        Args:
            trainer: Host runtime providing callbacks, placement, logging, and signals.
        """
        self._trainer = trainer

    def run(
        self,
        module: BasePhiModule,
        train_step: TrainStep,
        state: TrainState,
        batches: BatchSource,
        *,
        initial_step: int,
        balancer_update: _BalancerUpdateRuntime | None = None,
        hyperparameters: Mapping[str, Any] | None = None,
    ) -> FitResult:
        """Run compiled training updates over a callable or finite iterable batch source.

        Args:
            module: Application module owning numerical behavior and host lifecycle hooks.
            train_step: Compiled function mapping state and one batch to updated state and metrics.
            state: Initial functional training state and restore template.
            batches: Step-indexed callable or finite batch iterable.
            initial_step: Restored optimizer step copied to the host before source construction.
            balancer_update: Optional adaptive update and host-side schedule.
            hyperparameters: Optional resolved configuration recorded before the first update.

        Returns:
            Final state, metrics, stop status, and processed batch count.

        Raises:
            BaseException: Re-raises lifecycle, data-source, compiled-step, logging, or checkpoint errors after cleanup.
        """
        trainer = self._trainer
        metrics_connector = trainer._logger_connector
        signal_connector = trainer._signal_connector
        metrics_connector.set_metrics({})
        batch_iterator = None if callable(batches) else iter(batches)
        context = TrainerContext(
            state=state,
            step=initial_step,
            metrics={},
            module=module,
            is_global_zero=trainer.strategy.is_global_zero,
            has_logger=metrics_connector.has_logger,
        )
        module_context = PhiModuleContext(step=initial_step, metrics={})
        module_metrics: dict[str, Any] = {}
        module_logs: dict[str, _LoggedMetric] = {}
        callback_metrics: dict[str, Any] = {}
        final_metrics: dict[str, float] = {}
        stopped_early = False
        iterations = 0
        metrics_iteration = 0
        lifecycle = TaskLifecycle(
            trainer.callbacks,
            module,
            metrics_connector.logger,
            is_global_zero=trainer.strategy.is_global_zero,
        )
        signal_connector.reset()
        signal_connector.install()

        try:
            lifecycle.setup()
            if trainer._pending_callback_states is not None:
                trainer.load_callback_state_dict(trainer._pending_callback_states)
                trainer._pending_callback_states = None
            if trainer.strategy.is_global_zero:
                metrics_connector.logger.log_hyperparams(dict(hyperparameters or {}))
            for callback in trainer.callbacks:
                callback.on_fit_start(context)
            model_state = module.on_fit_start(state.model_state, module_context)
            state = replace(state, model_state=model_state)
            context = TrainerContext(
                state=state,
                step=initial_step,
                metrics={},
                module=module,
                is_global_zero=trainer.strategy.is_global_zero,
                has_logger=metrics_connector.has_logger,
            )

            for iteration in range(trainer.max_steps):
                try:
                    global_step = initial_step + iteration
                    batch = batches(global_step) if callable(batches) else next(batch_iterator)  # type: ignore[arg-type]
                except StopIteration:
                    break
                batch = trainer.prepare_batch(batch)
                context = TrainerContext(
                    state=state,
                    step=global_step,
                    metrics=dict(module_context.metrics),
                    module=module,
                    is_global_zero=trainer.strategy.is_global_zero,
                    batch=batch,
                    has_logger=metrics_connector.has_logger,
                )
                for callback in trainer.callbacks:
                    callback.on_train_batch_start(context)
                model_state, batch = module.on_train_batch_start(state.model_state, batch, module_context)
                state = replace(state, model_state=model_state)
                context = TrainerContext(
                    state=state,
                    step=global_step,
                    metrics=dict(module_context.metrics),
                    module=module,
                    is_global_zero=trainer.strategy.is_global_zero,
                    has_logger=metrics_connector.has_logger,
                )

                if balancer_update is not None:
                    update_due = (
                        global_step >= balancer_update.update_start_step
                        and (global_step - balancer_update.update_start_step) % balancer_update.every_n_steps == 0
                    )
                    if update_due:
                        update_batches = batch if balancer_update.batches is None else balancer_update.batches
                        balancer_state = balancer_update.update(
                            state.model_state,
                            update_batches,
                            state.balancer_state,
                        )
                        state = replace(state, balancer_state=balancer_state)
                state, device_metrics = train_step(state, batch)
                step = global_step + 1
                module_context = PhiModuleContext(step=step, metrics=dict(device_metrics))
                iterations += 1
                should_log = iterations == 1 or iterations % trainer.log_every_n_steps == 0
                context = TrainerContext(
                    state=state,
                    step=step,
                    metrics=dict(device_metrics),
                    module=module,
                    is_global_zero=trainer.strategy.is_global_zero,
                    should_log=should_log,
                    has_logger=metrics_connector.has_logger,
                )
                stop_requests = [callback.on_train_batch_end(context) for callback in trainer.callbacks]
                stopped_early = any(stop_requests)
                with _collect_module_metrics() as collected_metrics:
                    model_state, device_metrics = module.on_train_batch_end(state.model_state, module_context)
                    collected_metrics.add_defaults(device_metrics)
                state = replace(state, model_state=model_state)
                module_logs = dict(collected_metrics.records)
                device_metrics = {**device_metrics, **{name: record.value for name, record in module_logs.items()}}
                module_metrics = dict(device_metrics)
                module_context = PhiModuleContext(step=step, metrics=dict(device_metrics))
                context = TrainerContext(
                    state=state,
                    step=step,
                    metrics=dict(device_metrics),
                    module=module,
                    is_global_zero=trainer.strategy.is_global_zero,
                    should_log=should_log,
                    has_logger=metrics_connector.has_logger,
                )
                callback_metrics = metrics_connector.collect_callback_metrics(context, module_metrics)
                if callback_metrics:
                    device_metrics = {**device_metrics, **callback_metrics}
                    module_context = PhiModuleContext(step=step, metrics=dict(device_metrics))
                    context = TrainerContext(
                        state=state,
                        step=step,
                        metrics=dict(device_metrics),
                        module=module,
                        is_global_zero=trainer.strategy.is_global_zero,
                        should_log=should_log,
                        has_logger=metrics_connector.has_logger,
                    )

                metrics_connector.set_metrics(device_metrics, module_logs, tuple(callback_metrics))
                context = replace(context, callback_states=trainer.callback_state_dict())
                for callback in trainer.callbacks:
                    callback.on_train_metrics(context)

                if stopped_early or iteration == trainer.max_steps - 1:
                    jax.block_until_ready(device_metrics)
                    final_metrics = _scalar_metric_values(device_metrics)
                    metrics_iteration = iterations
                if should_log and trainer.strategy.is_global_zero and metrics_connector.logged_metrics:
                    metrics_connector.logger.log_metrics(scalar_metrics(metrics_connector.logged_metrics), step)
                if stopped_early:
                    break

            completed_step = initial_step + iterations
            state_step = _state_step(state)
            if state_step != completed_step:
                raise RuntimeError(
                    "A custom `TrainStep` must increment `TrainState.step` exactly once per completed update; "
                    f"expected step {completed_step}, got {state_step}."
                )
            context = TrainerContext(
                state=state,
                step=completed_step,
                metrics=module_metrics,
                module=module,
                is_global_zero=trainer.strategy.is_global_zero,
                is_fit_end=True,
                has_logger=metrics_connector.has_logger,
            )
            final_callback_metrics = metrics_connector.collect_callback_metrics(context, module_metrics)
            terminal_metrics = {**module_metrics, **callback_metrics, **final_callback_metrics}
            module_context = PhiModuleContext(step=context.step, metrics=terminal_metrics)
            context = TrainerContext(
                state=state,
                step=context.step,
                metrics=terminal_metrics,
                module=module,
                is_global_zero=trainer.strategy.is_global_zero,
                is_fit_end=True,
                has_logger=metrics_connector.has_logger,
            )
            callback_names = tuple({*callback_metrics, *final_callback_metrics})
            metrics_connector.set_metrics(terminal_metrics, module_logs, callback_names)
            context = replace(context, callback_states=trainer.callback_state_dict())
            if context.metrics and (metrics_iteration != iterations or final_callback_metrics):
                jax.block_until_ready(context.metrics)
                final_metrics = _scalar_metric_values(context.metrics)
            if final_callback_metrics and trainer.strategy.is_global_zero and metrics_connector.logged_metrics:
                metrics_connector.logger.log_metrics(scalar_metrics(metrics_connector.logged_metrics), context.step)
            for callback in trainer.callbacks:
                callback.on_fit_end(context)
            model_state = module.on_fit_end(state.model_state, module_context)
            state = replace(state, model_state=model_state)
            lifecycle.finalize("success")
            return FitResult(
                module=module,
                state=state,
                metrics=final_metrics,
                stopped_early=stopped_early,
                interrupted=False,
                iterations=iterations,
            )
        except KeyboardInterrupt as error:
            signal_connector.interrupted = True
            lifecycle.handle_exception(error, context, module_context)
            if context.metrics:
                jax.block_until_ready(context.metrics)
                final_metrics = _scalar_metric_values(context.metrics)
            if trainer.strategy.is_global_zero:
                log.warning(f"Training interrupted at step {context.step}; preserving the last completed state.")
                lifecycle.finalize("interrupted")
            return FitResult(
                module=module,
                state=context.state,
                metrics=final_metrics,
                stopped_early=False,
                interrupted=True,
                iterations=iterations,
            )
        except SystemExit as error:
            if not signal_connector.received_sigterm:
                lifecycle.handle_exception(error, context, module_context)
                lifecycle.finalize("failed")
                raise
            signal_connector.interrupted = True
            lifecycle.handle_exception(error, context, module_context)
            if context.metrics:
                jax.block_until_ready(context.metrics)
            if trainer.strategy.is_global_zero:
                log.warning(f"Training received SIGTERM at step {context.step}; terminating after checkpoint cleanup.")
            lifecycle.finalize("interrupted")
            raise
        except BaseException as error:
            lifecycle.handle_exception(error, context, module_context)
            lifecycle.finalize("failed")
            raise
        finally:
            signal_connector.restore()
            lifecycle.teardown()


def _state_step(state: TrainState) -> int:
    """Transfer the scalar optimizer step to a Python integer.

    Args:
        state: Functional training state.

    Returns:
        Host integer optimizer step.
    """
    return int(np.asarray(jax.device_get(state.step)))


def _scalar_metric_values(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Transfer only scalar entries from a complete diagnostic mapping.

    Args:
        metrics: Scalar metrics and optional array diagnostics.

    Returns:
        Host scalar mapping suitable for :class:`phijax.training.FitResult`.
    """
    return scalar_metrics({name: value for name, value in metrics.items() if _metric_is_scalar(value)})
