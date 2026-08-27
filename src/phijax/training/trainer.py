import logging
import signal
import threading
from collections.abc import Callable, Iterable, Mapping, Sequence, Sized
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from types import FrameType
from typing import Any, Self, cast

import jax
import numpy as np
import optax

from phijax.callbacks import Callback, ModelCheckpoint, PredictionContext, PredictionWriter, TrainerContext
from phijax.data import PhiDataModule
from phijax.module import BasePhiModule, PhiModuleContext
from phijax.training.checkpointing import restore_checkpoint
from phijax.training.lifecycle import TaskLifecycle
from phijax.training.loggers import ExperimentLogger, LoggerCollection, scalar_metrics
from phijax.training.plans import TrainingPlan
from phijax.training.precision import PrecisionPolicy
from phijax.training.state import TrainState, initialize_train_state
from phijax.training.steps import TrainStep, make_train_step, with_balancer_updates
from phijax.training.strategies import Strategy, create_strategy
from phijax.types import JaxDevice

type BatchSource = Iterable[Any] | Callable[[int], Any]

log = logging.getLogger(__name__)
_MATMUL_PRECISIONS = frozenset({"default", "high", "highest"})


@dataclass(frozen=True, slots=True)
class FitResult:
    """Summarize the terminal state of one trainer fit call.

    Attributes:
        state: Final functional training state.
        metrics: Final host scalar metrics.
        stopped_early: Whether a callback requested termination.
        iterations: Number of batches processed by this fit call.
        interrupted: Whether an operating-system signal or :class:`KeyboardInterrupt` stopped training.
    """

    state: TrainState
    metrics: dict[str, float]
    stopped_early: bool
    iterations: int
    interrupted: bool = False


class Trainer:
    """Coordinate compiled JAX updates, callbacks, logging, and checkpointing on the host.

    The trainer never owns a mutable model object and never places Python hooks inside :func:`jax.jit`. Numerical
    behavior remains in the configured training plan or supplied compiled step; this class orchestrates its lifecycle.

    Attributes:
        max_steps: Maximum number of batches processed by one fit call.
        precision: Resolved training precision policy.
        matmul_precision: Optional JAX dot and convolution precision override.
        strategy: Explicit device-placement strategy.
        prediction_writer: Configured prediction artifact callback, or `None`.
    """

    def __init__(
        self,
        max_steps: int,
        *,
        accelerator: str = "auto",
        devices: int | list[int] | str = 1,
        precision: str | PrecisionPolicy = "32-true",
        matmul_precision: str | None = None,
        derivative_dtype: Any | None = None,
        initial_loss_scale: float = 32768.0,
        loss_scale_growth_interval: int = 2000,
        deterministic: bool = True,
        log_every_n_steps: int = 10,
        callbacks: Iterable[Callback] = (),
        loggers: Iterable[ExperimentLogger] = (),
        strategy: Strategy | None = None,
        compilation_cache: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize host-side training orchestration.

        Args:
            max_steps: Maximum number of batches processed by one fit call.
            accelerator: Device backend selected when `strategy` is absent.
            devices: Device count, explicit indices, or `auto` when `strategy` is absent.
            precision: Lightning-compatible precision mode or resolved policy.
            matmul_precision: Optional `default`, `high`, or `highest` JAX matrix-multiplication precision override.
            derivative_dtype: Optional floating batch and coordinate-derivative dtype override.
            initial_loss_scale: Initial dynamic scale for `16-mixed` training.
            loss_scale_growth_interval: Finite optimizer updates between FP16 scale increases.
            deterministic: Whether the caller promises deterministic data and step behavior.
            log_every_n_steps: Positive interval between scalar logger updates.
            callbacks: Ordered host-side lifecycle callbacks.
            loggers: Logging backends receiving the same hyperparameters and metrics.
            strategy: Explicit placement strategy overriding `accelerator` and `devices`.
            compilation_cache: Optional mapping with `enabled` and `directory` values applied before device discovery.

        Raises:
            ValueError: If a step, interval, or matrix-multiplication precision option is invalid.
        """
        if max_steps < 1:
            raise ValueError("`max_steps` must be positive.")
        if log_every_n_steps < 1:
            raise ValueError("`log_every_n_steps` must be positive.")
        if matmul_precision is not None and matmul_precision not in _MATMUL_PRECISIONS:
            choices = ", ".join(sorted(_MATMUL_PRECISIONS))
            raise ValueError(f"`matmul_precision` must be `None` or one of: {choices}.")
        self.max_steps = max_steps
        self.precision = PrecisionPolicy.from_name(
            precision,
            derivative_dtype=derivative_dtype,
            initial_loss_scale=initial_loss_scale,
            growth_interval=loss_scale_growth_interval,
        )
        self.matmul_precision = matmul_precision
        cache_config = dict(compilation_cache or {})
        if cache_config.get("enabled", False):
            directory = cache_config.get("directory")
            if not directory:
                raise ValueError("Enabled `compilation_cache` requires a non-empty `directory`.")
            jax.config.update("jax_compilation_cache_dir", str(directory))
        self.strategy = strategy or create_strategy(accelerator, devices)
        self.deterministic = deterministic
        self.log_every_n_steps = log_every_n_steps
        self.callbacks = tuple(callbacks)
        checkpoint_callbacks = tuple(callback for callback in self.callbacks if isinstance(callback, ModelCheckpoint))
        if len(checkpoint_callbacks) > 1:
            raise ValueError("Only one `ModelCheckpoint` callback may be configured.")
        self.checkpoint_callback = checkpoint_callbacks[0] if checkpoint_callbacks else None
        prediction_writers = tuple(callback for callback in self.callbacks if isinstance(callback, PredictionWriter))
        if len(prediction_writers) > 1:
            raise ValueError("Only one `PredictionWriter` callback may be configured.")
        self.prediction_writer = prediction_writers[0] if prediction_writers else None
        self.logger = LoggerCollection(loggers)
        self.interrupted = False
        self.received_sigterm = False
        self._closed = False

    def print_environment_info(self) -> None:
        """Print Lightning-style precision and accelerator information on global rank zero."""
        if not self.strategy.is_global_zero:
            return

        precision_messages = {
            "64-true": "Using 64-bit true precision",
            "32-true": "Using 32-bit true precision",
            "16-true": "Using 16-bit true precision",
            "bf16-true": "Using bfloat16 true precision",
            "16-mixed": "Using 16-bit Automatic Mixed Precision (AMP)",
            "bf16-mixed": "Using bfloat16 Automatic Mixed Precision (AMP)",
        }
        print(flush=True)
        print(precision_messages[self.precision.mode], flush=True)
        resolved_matmul_precision = self.matmul_precision or jax.config.jax_default_matmul_precision or "default"
        print(f"Matmul precision: {resolved_matmul_precision}", flush=True)

        selected_devices = self.strategy.devices
        # The default backend and explicitly selected backend may differ when the user forces CPU execution.
        visible_devices = _unique_devices((*jax.devices(), *selected_devices))
        available_gpus = tuple(device for device in visible_devices if device.platform in {"cuda", "gpu", "rocm"})
        selected_gpus = tuple(device for device in selected_devices if device.platform in {"cuda", "gpu", "rocm"})
        gpu_backend = _gpu_backend_name((*available_gpus, *selected_gpus))
        backend_suffix = f" ({gpu_backend})" if available_gpus else ""
        print(f"GPU available: {bool(available_gpus)}{backend_suffix}, used: {bool(selected_gpus)}", flush=True)

        available_tpus = tuple(device for device in visible_devices if device.platform == "tpu")
        selected_tpus = tuple(device for device in selected_devices if device.platform == "tpu")
        print(f"TPU available: {bool(available_tpus)}, using: {len(selected_tpus)} TPU cores", flush=True)
        print(flush=True)

    def initialize_state(
        self,
        model_state: Any,
        optimizer: optax.GradientTransformation,
        balancer_state: Any,
        rng_key: jax.Array,
    ) -> TrainState:
        """Initialize functional state with this trainer's precision policy.

        Args:
            model_state: Explicit differentiable model parameter PyTree.
            optimizer: Optax transformation used to initialize optimizer slots.
            balancer_state: Initial loss-balancer state.
            rng_key: Explicit training PRNG key returned or derived from :func:`phijax.utils.seed_everything`.

        Returns:
            Complete precision-aware training state.
        """
        return initialize_train_state(model_state, optimizer, balancer_state, rng_key, self.precision)

    def compile_train_step(
        self,
        module: BasePhiModule,
        balancer: Any,
        optimizer: optax.GradientTransformation,
    ) -> TrainStep:
        """Build a compiled update with this trainer's precision policy.

        Args:
            module: Application module exposing named unweighted losses.
            balancer: Functional loss balancer.
            optimizer: Optax gradient transformation used by initialized state.

        Returns:
            Reusable JIT-compiled training step.
        """
        return make_train_step(module, balancer, optimizer, self.precision)

    def _collect_callback_metrics(
        self,
        context: TrainerContext,
        existing_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Collect and validate metrics contributed by configured callbacks.

        Args:
            context: Current post-module trainer context.
            existing_metrics: Metrics already produced by the compiled step and module.

        Returns:
            Uniquely named callback metrics in callback declaration order.

        Raises:
            TypeError: If a callback does not return a mapping.
            ValueError: If a callback returns an invalid or colliding metric name.
        """
        callback_metrics: dict[str, Any] = {}
        for callback in self.callbacks:
            contributed_metrics = callback.training_metrics(context)
            if not isinstance(contributed_metrics, Mapping):
                raise TypeError(f"{type(callback).__name__}.training_metrics() must return a mapping.")
            invalid_names = tuple(name for name in contributed_metrics if not isinstance(name, str) or not name.strip())
            if invalid_names:
                raise ValueError(f"Callback metric names must be non-empty strings: {invalid_names}.")
            existing_names = set(existing_metrics) | set(callback_metrics)
            collisions = existing_names & set(contributed_metrics)
            if collisions:
                raise ValueError(f"Callback metrics collide with existing names: {sorted(collisions)}.")
            callback_metrics.update(contributed_metrics)
        return callback_metrics

    def prepare_batch_source[SourceT](self, source: SourceT) -> SourceT:
        """Prepare persistent sampler state for this Trainer's process-local root device.

        Sources without a callable `prepare` method are returned unchanged, preserving support for ordinary callables
        and finite iterables. Built-in PhiJAX sources transfer finite candidate pools, generation bounds, templates,
        and PRNG keys once instead of transferring sampled training batches from the host on every step.

        Args:
            source: Training batch callable or iterable, optionally exposing `prepare(device)`.

        Returns:
            Source prepared for the selected Strategy, or the original source when preparation is unsupported.
        """
        prepare = getattr(source, "prepare", None)
        if not callable(prepare):
            return source
        return cast(SourceT, prepare(self.strategy.root_device))

    def prepare_batch(self, batch: Any) -> Any:
        """Cast and place one batch according to the configured precision and Strategy.

        NumPy floating leaves are cast on the host before transfer. Device-resident sampler outputs retain their
        placement when the Strategy target is unchanged, while data-parallel Strategies apply final batch sharding.

        Args:
            batch: Arbitrary host or device batch PyTree.

        Returns:
            Precision-normalized batch placed or sharded for compiled execution.
        """
        return self.strategy.place_batch(self.precision.cast_batch(batch))

    def fit(
        self,
        module: BasePhiModule,
        training: TrainingPlan | TrainStep,
        state: TrainState,
        batches: BatchSource | None = None,
        *,
        hyperparameters: Mapping[str, Any] | None = None,
        ckpt_path: str | Path | None = None,
        ckpt_step: int | None = None,
        weights_only: bool = False,
        datamodule: PhiDataModule | None = None,
        sampling_key: jax.Array | None = None,
        balancer_key: jax.Array | None = None,
    ) -> FitResult:
        """Run training under the configured matrix-multiplication precision policy.

        Args:
            module: Application module owning model, objective, and lifecycle behavior.
            training: Configured training plan, or a compiled step when `batches` is supplied explicitly.
            state: Initial functional training state and restore template.
            batches: Optional explicit step-indexed callable or finite iterable batch source. When omitted, the
                Trainer requests a source from `datamodule` using the plan's batch keys.
            hyperparameters: Optional resolved configuration recorded before the first update.
            ckpt_path: Optional Orbax checkpoint root used for resumption or weight initialization.
            ckpt_step: Exact checkpoint step, or `None` for the latest committed step.
            weights_only: Whether to restore only model weights into the fresh state.
            datamodule: Optional DataModule prepared, queried, and torn down for the fit stage.
            sampling_key: Explicit source-sampling key required when `batches` is omitted.
            balancer_key: Explicit diagnostic-sampling key required by adaptive plans with fixed diagnostic batches.

        Returns:
            Final state, metrics, stop status, and processed batch count.

        Raises:
            ValueError: If checkpoint restoration or DataModule source options are incomplete or invalid.
            BaseException: Re-raises training lifecycle errors after cleanup.
        """
        try:
            plan = training if isinstance(training, TrainingPlan) else TrainingPlan(training)
            batches = self._training_batch_source(plan, batches, datamodule, sampling_key)
            train_step = self._training_step(plan, batches, balancer_key)
            state = restore_checkpoint(
                state,
                ckpt_path,
                weights_only=weights_only,
                step=ckpt_step,
            )
            precision_context = (
                nullcontext() if self.matmul_precision is None else jax.default_matmul_precision(self.matmul_precision)
            )
            with precision_context:
                return self._fit(
                    module,
                    train_step,
                    state,
                    batches,
                    hyperparameters=hyperparameters,
                )
        finally:
            if datamodule is not None:
                datamodule.teardown_stage("fit")

    def _training_batch_source(
        self,
        plan: TrainingPlan,
        batches: BatchSource | None,
        datamodule: PhiDataModule | None,
        sampling_key: jax.Array | None,
    ) -> BatchSource:
        """Resolve and place an explicit or DataModule-owned training source.

        Args:
            plan: Training plan declaring required named batch keys.
            batches: Optional explicit training source.
            datamodule: Optional DataModule owning the default source.
            sampling_key: Explicit root sampling key for a DataModule-owned source.

        Returns:
            Source prepared for the Trainer's process-local root device.

        Raises:
            ValueError: If no explicit source is supplied and DataModule requirements are incomplete.
        """
        if batches is not None:
            return self.prepare_batch_source(batches)
        if datamodule is None:
            raise ValueError("`Trainer.fit()` requires either explicit `batches` or a `datamodule`.")
        if not plan.batch_keys:
            raise ValueError("A DataModule-owned training source requires non-empty `TrainingPlan.batch_keys`.")
        if sampling_key is None:
            raise ValueError("A DataModule-owned training source requires an explicit `sampling_key`.")
        datamodule.prepare_stage("fit")
        return self.prepare_batch_source(datamodule.train_batch_source(plan.batch_keys, sampling_key))

    def _training_step(
        self,
        plan: TrainingPlan,
        batches: BatchSource,
        balancer_key: jax.Array | None,
    ) -> TrainStep:
        """Bind optional adaptive-balancer diagnostics to a configured training step.

        Args:
            plan: Training plan containing the compiled optimizer step and optional update schedule.
            batches: Prepared training source used for fixed diagnostic sampling when requested.
            balancer_key: Optional explicit diagnostic sampling key.

        Returns:
            Compiled step or host-scheduled adaptive wrapper ready for the fit loop.

        Raises:
            TypeError: If fixed diagnostics are requested from a source without explicit sampling support.
            ValueError: If fixed diagnostic sampling has no `balancer_key`.
        """
        schedule = plan.balancer_update
        if schedule is None:
            return plan.train_step
        update_batches = None
        if schedule.plan.batch_sizes is not None:
            if balancer_key is None:
                raise ValueError("Fixed adaptive-balancer diagnostics require an explicit `balancer_key`.")
            sample = getattr(batches, "sample", None)
            if not callable(sample):
                raise TypeError("Fixed adaptive-balancer diagnostics require a training source with `sample()`.")
            update_batches = self.prepare_batch(sample(balancer_key, schedule.plan.batch_sizes))
        return with_balancer_updates(
            plan.train_step,
            schedule.plan.update,
            update_batches,
            schedule.every_n_steps,
            skip_first_step=schedule.skip_first_step,
        )

    def _fit(
        self,
        module: BasePhiModule,
        train_step: TrainStep,
        state: TrainState,
        batches: BatchSource,
        *,
        hyperparameters: Mapping[str, Any] | None = None,
    ) -> FitResult:
        """Run compiled training updates over a callable or finite iterable batch source.

        Args:
            module: Application module owning model and objective behavior plus overridable host lifecycle hooks.
            train_step: Compiled function mapping state and one batch to updated state and scalar metrics.
            state: Initial functional training state and restore template.
            batches: Callable accepting the global optimizer step, including the restored state offset, or a finite
                batch iterable.
            hyperparameters: Optional resolved configuration recorded before the first update.

        Returns:
            Final state, metrics, stop status, and processed batch count.

        Raises:
            BaseException: Re-raises any module, callback, data-source, compiled-step, logging, or checkpoint error
                after exception hooks and logger finalization run.
        """
        state = self.strategy.place_state(state)
        batch_iterator = None if callable(batches) else iter(batches)
        initial_step = _state_step(state)
        context = TrainerContext(
            state=state,
            step=initial_step,
            metrics={},
            module=module,
            is_global_zero=self.strategy.is_global_zero,
        )
        module_context = PhiModuleContext(step=initial_step, metrics={})
        module_metrics: dict[str, Any] = {}
        callback_metrics: dict[str, Any] = {}
        final_metrics: dict[str, float] = {}
        stopped_early = False
        iterations = 0
        metrics_iteration = 0
        lifecycle = TaskLifecycle(
            self.callbacks,
            module,
            self.logger,
            is_global_zero=self.strategy.is_global_zero,
        )
        self.interrupted = False
        self.received_sigterm = False
        signal_handler = _FitSignalHandler(self)
        signal_handler.install()

        try:
            lifecycle.setup()
            if self.strategy.is_global_zero:
                self.logger.log_hyperparameters(dict(hyperparameters or {}))
            for callback in self.callbacks:
                callback.on_fit_start(context)
            model_state = module.on_fit_start(state.model_state, module_context)
            state = replace(state, model_state=model_state)
            context = TrainerContext(
                state=state,
                step=initial_step,
                metrics={},
                module=module,
                is_global_zero=self.strategy.is_global_zero,
            )

            for iteration in range(self.max_steps):
                try:
                    global_step = initial_step + iteration
                    batch = batches(global_step) if callable(batches) else next(batch_iterator)  # type: ignore[arg-type]
                except StopIteration:
                    break
                batch = self.prepare_batch(batch)
                for callback in self.callbacks:
                    callback.on_train_batch_start(context)
                model_state, batch = module.on_train_batch_start(state.model_state, batch, module_context)
                state = replace(state, model_state=model_state)
                context = TrainerContext(
                    state=state,
                    step=_state_step(state),
                    metrics=dict(module_context.metrics),
                    module=module,
                    is_global_zero=self.strategy.is_global_zero,
                )

                state, device_metrics = train_step(state, batch)
                step = _state_step(state)
                module_context = PhiModuleContext(step=step, metrics=dict(device_metrics))
                iterations += 1
                should_log = iterations == 1 or iterations % self.log_every_n_steps == 0
                context = TrainerContext(
                    state=state,
                    step=step,
                    metrics=dict(device_metrics),
                    module=module,
                    is_global_zero=self.strategy.is_global_zero,
                    should_log=should_log,
                )
                stop_requests = [callback.on_train_batch_end(context) for callback in self.callbacks]
                stopped_early = any(stop_requests)
                model_state, device_metrics = module.on_train_batch_end(state.model_state, module_context)
                state = replace(state, model_state=model_state)
                module_metrics = dict(device_metrics)
                module_context = PhiModuleContext(step=step, metrics=dict(device_metrics))
                context = TrainerContext(
                    state=state,
                    step=step,
                    metrics=dict(device_metrics),
                    module=module,
                    is_global_zero=self.strategy.is_global_zero,
                    should_log=should_log,
                )
                callback_metrics = self._collect_callback_metrics(context, module_metrics)
                if callback_metrics:
                    device_metrics = {**device_metrics, **callback_metrics}
                    module_context = PhiModuleContext(step=step, metrics=dict(device_metrics))
                    context = TrainerContext(
                        state=state,
                        step=step,
                        metrics=dict(device_metrics),
                        module=module,
                        is_global_zero=self.strategy.is_global_zero,
                        should_log=should_log,
                    )

                if should_log or stopped_early or iteration == self.max_steps - 1:
                    jax.block_until_ready(device_metrics)
                    final_metrics = scalar_metrics(device_metrics)
                    metrics_iteration = iterations
                if should_log and self.strategy.is_global_zero:
                    self.logger.log_metrics(final_metrics, step)
                if stopped_early:
                    break

            context = TrainerContext(
                state=state,
                step=_state_step(state),
                metrics=module_metrics,
                module=module,
                is_global_zero=self.strategy.is_global_zero,
                is_fit_end=True,
            )
            final_callback_metrics = self._collect_callback_metrics(context, module_metrics)
            terminal_metrics = {**module_metrics, **callback_metrics, **final_callback_metrics}
            module_context = PhiModuleContext(step=context.step, metrics=terminal_metrics)
            context = TrainerContext(
                state=state,
                step=context.step,
                metrics=terminal_metrics,
                module=module,
                is_global_zero=self.strategy.is_global_zero,
                is_fit_end=True,
            )
            if context.metrics and (metrics_iteration != iterations or final_callback_metrics):
                jax.block_until_ready(context.metrics)
                final_metrics = scalar_metrics(context.metrics)
            if final_callback_metrics and self.strategy.is_global_zero:
                self.logger.log_metrics(final_metrics, context.step)
            for callback in self.callbacks:
                callback.on_fit_end(context)
            model_state = module.on_fit_end(state.model_state, module_context)
            state = replace(state, model_state=model_state)
            context = TrainerContext(
                state=state,
                step=_state_step(state),
                metrics=dict(module_context.metrics),
                module=module,
                is_global_zero=self.strategy.is_global_zero,
                is_fit_end=True,
            )
            lifecycle.finalize("success")
            return FitResult(
                state=state,
                metrics=final_metrics,
                stopped_early=stopped_early,
                interrupted=False,
                iterations=iterations,
            )
        except KeyboardInterrupt as error:
            self.interrupted = True
            lifecycle.handle_exception(error, context, module_context)
            if context.metrics:
                jax.block_until_ready(context.metrics)
                final_metrics = scalar_metrics(context.metrics)
            if self.strategy.is_global_zero:
                log.warning(f"Training interrupted at step {context.step}; preserving the last completed state.")
                lifecycle.finalize("interrupted")
            return FitResult(
                state=context.state,
                metrics=final_metrics,
                stopped_early=False,
                interrupted=True,
                iterations=iterations,
            )
        except SystemExit as error:
            if not self.received_sigterm:
                lifecycle.handle_exception(error, context, module_context)
                lifecycle.finalize("failed")
                raise
            self.interrupted = True
            lifecycle.handle_exception(error, context, module_context)
            if context.metrics:
                jax.block_until_ready(context.metrics)
            if self.strategy.is_global_zero:
                log.warning(f"Training received SIGTERM at step {context.step}; terminating after checkpoint cleanup.")
            lifecycle.finalize("interrupted")
            raise
        except BaseException as error:
            lifecycle.handle_exception(error, context, module_context)
            lifecycle.finalize("failed")
            raise
        finally:
            signal_handler.restore()
            lifecycle.teardown()

    def resume_latest(
        self,
        module: BasePhiModule,
        training: TrainingPlan | TrainStep,
        state: TrainState,
        batches: BatchSource | None = None,
        *,
        hyperparameters: Mapping[str, Any] | None = None,
        datamodule: PhiDataModule | None = None,
        sampling_key: jax.Array | None = None,
        balancer_key: jax.Array | None = None,
    ) -> FitResult:
        """Restore the latest full checkpoint and continue fitting.

        Args:
            module: Application module supplied to :meth:`fit` after restoration.
            training: Configured training plan, or compiled step paired with explicit `batches`.
            state: Restore template matching the checkpoint structure.
            batches: Optional explicit callable or finite iterable batch source.
            hyperparameters: Optional resolved configuration logged before updates.
            datamodule: Optional DataModule supplying the default source.
            sampling_key: Explicit DataModule source-sampling key.
            balancer_key: Explicit adaptive-balancer diagnostic key.

        Returns:
            Result from the resumed fit call.

        Raises:
            ValueError: If checkpointing is not configured.
            FileNotFoundError: If no checkpoint is available.
        """
        if self.checkpoint_callback is None:
            raise ValueError("`resume_latest()` requires a configured `ModelCheckpoint` callback.")
        latest_step = self.checkpoint_callback.checkpoint_io.latest_step
        if latest_step is None:
            raise FileNotFoundError("No checkpoint is available for resumption.")
        state = self.checkpoint_callback.checkpoint_io.restore(state, latest_step)
        return self.fit(
            module,
            training,
            state,
            batches,
            hyperparameters=hyperparameters,
            datamodule=datamodule,
            sampling_key=sampling_key,
            balancer_key=balancer_key,
        )

    def predict(
        self,
        module: BasePhiModule,
        state: TrainState,
        batches: Iterable[Mapping[str, Any]] | None = None,
        *,
        ckpt_path: str | Path | None = None,
        ckpt_step: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        return_predictions: bool = True,
        datamodule: PhiDataModule | None = None,
    ) -> np.ndarray | None:
        """Restore optional checkpoint weights and run prediction under the configured precision policy.

        This follows Lightning's `ckpt_path` convention while preserving PhiJAX's explicit functional state. The
        supplied `state` defines the complete Orbax restore structure; only its model state is replaced for prediction.
        When `batches` is omitted, the trainer prepares and sets up `datamodule` for prediction and requests its batch
        source. Prediction is skipped when neither source is available.

        Args:
            module: Application module exposing prediction computation and lifecycle hooks.
            state: Fresh or in-memory functional state supplying the model and optional restore template.
            batches: Optional finite prediction batch iterable. When omitted, use `datamodule`.
            ckpt_path: Optional Orbax checkpoint root. `None` predicts from the supplied in-memory state.
            ckpt_step: Exact checkpoint step, or `None` for the latest committed step.
            metadata: Optional immutable metadata exposed to prediction callbacks.
            return_predictions: Whether to concatenate and return host predictions.
            datamodule: Optional DataModule set up when `batches` is omitted and torn down after prediction.

        Returns:
            Concatenated host predictions, or `None` when collection is disabled or no prediction data is available.

        Raises:
            ValueError: If checkpoint options are invalid or a provided batch source is empty.
            BaseException: Re-raises prediction lifecycle errors after cleanup.
        """
        try:
            if batches is None:
                if datamodule is None:
                    return None
                datamodule.prepare_stage("predict")
                batches = datamodule.predict_batch_source()
                if batches is None:
                    return None
            restored_state = restore_checkpoint(
                state,
                ckpt_path,
                weights_only=ckpt_path is not None,
                step=ckpt_step,
            )
            precision_context = (
                nullcontext() if self.matmul_precision is None else jax.default_matmul_precision(self.matmul_precision)
            )
            with precision_context:
                return self._predict(
                    module,
                    restored_state.model_state,
                    batches,
                    metadata=metadata,
                    return_predictions=return_predictions,
                )
        finally:
            if datamodule is not None:
                datamodule.teardown_stage("predict")

    def _predict(
        self,
        module: BasePhiModule,
        model_state: Any,
        batches: Iterable[Mapping[str, Any]],
        *,
        metadata: Mapping[str, Any] | None = None,
        return_predictions: bool = True,
    ) -> np.ndarray | None:
        """Run fixed-size prediction batches with callback lifecycle dispatch.

        Callbacks run before matching module hooks. Batch-end hooks receive each valid, unpadded output, while epoch-end
        and predict-end hooks receive the final concatenated host array when collection is enabled.

        Args:
            module: Application module exposing :meth:`BasePhiModule.predict_step`.
            model_state: Explicit restored model parameter PyTree.
            batches: Finite iterable of prediction batches, optionally containing a Boolean `mask`.
            metadata: Optional immutable task metadata exposed to prediction callbacks.
            return_predictions: Whether to collect and concatenate batch predictions on the host. Set to `False` when
                prediction callbacks stream results to external storage.

        Returns:
            Concatenated host predictions in batch iteration order, or `None` when `return_predictions` is `False`.

        Raises:
            ValueError: If `batches` is empty.
            BaseException: Re-raises prediction or callback failures after exception hooks and finalization.
        """
        model_state = self.strategy.place_state(model_state)
        task_metadata = dict(metadata or {})
        prediction_pool = getattr(batches, "pool", None)
        total_batches = len(batches) if isinstance(batches, Sized) else None
        context = PredictionContext(
            outputs=None,
            batch_index=None,
            metadata=task_metadata,
            total_batches=total_batches,
            pool=prediction_pool,
            is_global_zero=self.strategy.is_global_zero,
        )
        module_context = PhiModuleContext(step=0, metrics={})
        lifecycle = TaskLifecycle(
            self.callbacks,
            module,
            self.logger,
            is_global_zero=self.strategy.is_global_zero,
        )
        outputs: list[np.ndarray] = []
        batch_count = 0
        try:
            lifecycle.setup()
            for callback in self.callbacks:
                callback.on_predict_start(context)
            module.on_predict_start(model_state, context)
            for callback in self.callbacks:
                callback.on_predict_epoch_start(context)
            module.on_predict_epoch_start(model_state, context)
            for batch_index, batch in enumerate(batches):
                placed_batch = self.prepare_batch(batch)
                context = PredictionContext(
                    outputs=None,
                    batch=placed_batch,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    metadata=task_metadata,
                    pool=prediction_pool,
                    is_global_zero=self.strategy.is_global_zero,
                )
                for callback in self.callbacks:
                    callback.on_predict_batch_start(context)
                module.on_predict_batch_start(model_state, context)
                prediction = module.predict_step(model_state, placed_batch)
                mask = placed_batch.get("mask")
                if mask is not None:
                    prediction = prediction[mask]
                batch_count += 1
                context = PredictionContext(
                    outputs=prediction,
                    batch=placed_batch,
                    batch_index=batch_index,
                    total_batches=total_batches,
                    metadata=task_metadata,
                    pool=prediction_pool,
                    is_global_zero=self.strategy.is_global_zero,
                )
                for callback in self.callbacks:
                    callback.on_predict_batch_end(context)
                module.on_predict_batch_end(model_state, context)
                if return_predictions:
                    # Match Lightning's prediction loop by releasing accelerator outputs after every batch.
                    outputs.append(np.asarray(jax.device_get(prediction)))
            if batch_count == 0:
                raise ValueError("Prediction requires at least one batch.")
            combined = np.concatenate(outputs, axis=0) if return_predictions else None
            context = PredictionContext(
                outputs=combined,
                batch_index=None,
                total_batches=total_batches,
                metadata=task_metadata,
                pool=prediction_pool,
                is_global_zero=self.strategy.is_global_zero,
            )
            for callback in self.callbacks:
                callback.on_predict_epoch_end(context)
            module.on_predict_epoch_end(model_state, context)
            for callback in self.callbacks:
                callback.on_predict_end(context)
            module.on_predict_end(model_state, context)
            lifecycle.finalize("success")
            return combined
        except BaseException as error:
            lifecycle.handle_exception(error, context, module_context)
            lifecycle.finalize("failed")
            raise
        finally:
            lifecycle.teardown()

    def load_weights(self, state: TrainState, step: int | None = None) -> TrainState:
        """Load model weights while preserving fresh optimizer and run state.

        Args:
            state: Fresh target training state.
            step: Checkpoint step, or `None` for the latest checkpoint.

        Returns:
            Fresh state with restored model weights.

        Raises:
            ValueError: If checkpointing is not configured.
        """
        if self.checkpoint_callback is None:
            raise ValueError("`load_weights()` requires a configured `ModelCheckpoint` callback.")
        return self.checkpoint_callback.checkpoint_io.restore_weights(state, step)

    def latest_checkpoint(self) -> tuple[Path, int]:
        """Resolve the latest committed filesystem checkpoint.

        Returns:
            Checkpoint-root path and latest committed step.

        Raises:
            FileNotFoundError: If the checkpoint backend contains no committed state.
            ValueError: If checkpointing is disabled or its backend has no filesystem directory.
        """
        if self.checkpoint_callback is None:
            raise ValueError("Resolving the latest checkpoint requires a configured `ModelCheckpoint` callback.")
        checkpoint_io = self.checkpoint_callback.checkpoint_io
        checkpoint_step = checkpoint_io.latest_step
        if checkpoint_step is None:
            raise FileNotFoundError("The configured checkpoint backend contains no committed state.")
        checkpoint_directory = getattr(checkpoint_io, "directory", None)
        if checkpoint_directory is None:
            raise ValueError("The configured checkpoint backend does not expose a filesystem `directory`.")
        return Path(checkpoint_directory).expanduser().resolve(), checkpoint_step

    def close(self) -> None:
        """Wait for pending checkpoint writes and release manager resources."""
        if self._closed:
            return
        self._closed = True
        if self.checkpoint_callback is not None:
            self.checkpoint_callback.close()

    def __enter__(self) -> Self:
        """Enter a trainer resource context.

        Returns:
            This trainer instance.
        """
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        """Release trainer resources when leaving a context.

        Args:
            exception_type: Active exception type, if any.
            exception: Active exception instance, if any.
            traceback: Active traceback, if any.
        """
        del exception_type, exception, traceback
        self.close()


class _FitSignalHandler:
    """Convert process interruption signals into a trainer-owned graceful shutdown."""

    def __init__(self, trainer: Trainer) -> None:
        """Initialize signal bookkeeping for one fit call.

        Args:
            trainer: Trainer whose interruption state is updated by received signals.
        """
        self._trainer = trainer
        self._previous: dict[int, Any] = {}
        self._received = 0

    def install(self) -> None:
        """Install `SIGINT` and `SIGTERM` handlers when running on the main Python thread."""
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)

    def restore(self) -> None:
        """Restore every process signal handler replaced by :meth:`install`."""
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Raise an interruption on the first signal and escalate repeated signals.

        Args:
            signum: Operating-system signal number.
            frame: Current interpreter frame supplied by :mod:`signal`.

        Raises:
            KeyboardInterrupt: On the first `SIGINT`.
            SystemExit: On the first `SIGTERM`, or a repeated signal whose previous handler cannot be called.
        """
        self._received += 1
        if signum == signal.SIGTERM:
            self._trainer.received_sigterm = True
        if self._received == 1:
            if signum == signal.SIGTERM:
                raise SystemExit(128 + signum)
            signal_name = signal.Signals(signum).name
            raise KeyboardInterrupt(f"Received {signal_name}.")

        previous = self._previous.get(signum, signal.SIG_DFL)
        signal.signal(signum, previous)
        if callable(previous):
            previous(signum, frame)
            return
        raise SystemExit(128 + signum)


def _unique_devices(devices: Sequence[JaxDevice]) -> tuple[JaxDevice, ...]:
    """Remove repeated JAX devices while preserving discovery order.

    Args:
        devices: Devices discovered from the default and selected backends.

    Returns:
        Devices unique by platform, process, and local identifier.
    """
    unique: dict[tuple[str, int, int], JaxDevice] = {}
    for device in devices:
        identity = (device.platform, device.process_index, device.id)
        unique.setdefault(identity, device)
    return tuple(unique.values())


def _gpu_backend_name(devices: Sequence[JaxDevice]) -> str:
    """Infer the user-facing GPU runtime name from JAX device metadata.

    Args:
        devices: Available or selected GPU devices.

    Returns:
        `cuda`, `rocm`, or the generic `gpu` label.
    """
    descriptions = " ".join(
        f"{device.platform} {getattr(device, 'platform_version', '')} "
        f"{getattr(getattr(device, 'client', None), 'platform_version', '')} "
        f"{getattr(device, 'device_kind', '')}".lower()
        for device in devices
    )
    if "rocm" in descriptions or "amd" in descriptions:
        return "rocm"
    if "cuda" in descriptions or "nvidia" in descriptions:
        return "cuda"
    return "gpu"


def _state_step(state: TrainState) -> int:
    """Transfer the scalar optimizer step to a Python integer.

    Args:
        state: Functional training state.

    Returns:
        Host integer optimizer step.
    """
    return int(np.asarray(jax.device_get(state.step)))


__all__ = ["BatchSource", "FitResult", "TrainStep", "Trainer"]
