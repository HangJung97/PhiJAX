import json
import warnings
from collections.abc import Callable, Collection, Generator, Iterable, Mapping, Sequence
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import replace
from pathlib import Path
from typing import Any, Self, cast

import jax
import numpy as np
import optax

from phijax.balancers import AdaptiveBalancer, BalancerUpdatePlan, LossBalancer, StaticLossBalancer
from phijax.callbacks import (
    Callback,
    CheckpointIO,
    ModelCheckpoint,
    ModelSummary,
    PredictionWriter,
    ProgressBar,
    TQDMProgressBar,
)
from phijax.core import BasePhiModule
from phijax.data import DataStage, PhiDataModule
from phijax.training.checkpointing import LegacyCheckpointError, OrbaxCheckpointIO
from phijax.training.connectors.logger_connector import _LoggerConnector
from phijax.training.connectors.signal_connector import _SignalConnector
from phijax.training.loggers import ExperimentLogger, LoggerCollection
from phijax.training.loops.fit_loop import _FitLoop, _state_step
from phijax.training.loops.prediction_loop import _PredictionLoop
from phijax.training.precision import MatmulPrecision, PrecisionName, PrecisionPolicy, configure_precision
from phijax.training.results import FitResult
from phijax.training.state import TrainState, as_input_statistics, initialize_train_state
from phijax.training.steps import TrainStep, make_train_step
from phijax.training.strategies import Accelerator, DeviceSelection, Strategy, create_strategy
from phijax.types import JaxDevice

type BatchSource = Iterable[Any] | Callable[[int], Any]

_MATMUL_PRECISIONS = frozenset({"default", "high", "highest"})


class Trainer:
    """Coordinate compiled JAX updates, callbacks, logging, and checkpointing on the host.

    The trainer never owns a mutable model object and never places Python hooks inside :func:`jax.jit`. Numerical
    behavior remains in the module and its compiled train step; this class orchestrates their lifecycle.

    Attributes:
        prediction_writer: Configured prediction artifact callback, or `None`.
        should_stop: Whether a callback has requested that fitting stop. The fit loop resets it when a fit starts and
            checks it before each step and after each step's `on_train_metrics` hooks, as in Lightning.
    """

    def __init__(
        self,
        max_steps: int,
        *,
        accelerator: Accelerator = "auto",
        devices: DeviceSelection = 1,
        precision: PrecisionName | PrecisionPolicy = "32-true",
        matmul_precision: MatmulPrecision | None = None,
        derivative_dtype: Any | None = None,
        initial_loss_scale: float | None = None,
        loss_scale_growth_interval: int | None = None,
        log_every_n_steps: int = 10,
        enable_progress_bar: bool = True,
        enable_model_summary: bool = True,
        callbacks: Iterable[Callback] = (),
        logger: bool | ExperimentLogger | Iterable[ExperimentLogger] | None = True,
        default_root_dir: str | Path = ".",
        loggers: Iterable[ExperimentLogger] | None = None,
        strategy: Strategy | None = None,
        compilation_cache: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize host-side training orchestration.

        Args:
            max_steps: Global optimizer step at which training stops. A fit resumed from a checkpoint runs only the
                remaining `max_steps - step` updates.
            accelerator: Device backend selected when `strategy` is absent. Choose `auto`, `cpu`, `gpu`, or `tpu`.
            devices: Positive device count per process, explicit local device indices, or `auto` for all matching
                devices. A multi-process run selects the same local devices on every process.
            precision: Precision name or resolved policy. Supported names are `64-true`, `32-true`, `16-true`,
                `bf16-true`, `16-mixed`, and `bf16-mixed`; common dtype aliases are also accepted. The name `64-true`
                enables JAX x64 for the whole process. A resolved policy cannot be combined with `derivative_dtype`,
                `initial_loss_scale`, or `loss_scale_growth_interval`.
            matmul_precision: Optional `default`, `high`, or `highest` JAX matrix-multiplication precision override.
            derivative_dtype: Optional floating batch and coordinate-derivative dtype override.
            initial_loss_scale: Initial dynamic scale for `16-mixed` training. `None` uses `32768.0`.
            loss_scale_growth_interval: Finite optimizer updates between FP16 scale increases. `None` uses `2000`.
            log_every_n_steps: Positive interval between scalar logger updates.
            enable_progress_bar: Whether to enable a TQDM or explicitly configured progress callback.
            enable_model_summary: Whether to add a plain model summary when none is supplied explicitly.
            callbacks: Ordered host-side lifecycle callbacks.
            logger: Default logger flag, one backend, several backends, or `None` to disable logging.
            default_root_dir: Parent directory for versioned default local logs.
            loggers: Deprecated plural alias for an explicit iterable of logging backends.
            strategy: Explicit placement strategy overriding `accelerator` and `devices`.
            compilation_cache: Optional mapping with `enabled` and `directory` values applied before device discovery.

        Raises:
            TypeError: If a display flag or logger configuration has an invalid type.
            ValueError: If a step, interval, callback, logger, precision, or matrix-multiplication precision option is
                invalid.
        """
        if max_steps < 1:
            raise ValueError("`max_steps` must be positive.")
        if log_every_n_steps < 1:
            raise ValueError("`log_every_n_steps` must be positive.")
        if not isinstance(enable_progress_bar, bool):
            raise TypeError("`enable_progress_bar` must be a boolean.")
        if not isinstance(enable_model_summary, bool):
            raise TypeError("`enable_model_summary` must be a boolean.")
        if matmul_precision is not None and matmul_precision not in _MATMUL_PRECISIONS:
            choices = ", ".join(sorted(_MATMUL_PRECISIONS))
            raise ValueError(f"`matmul_precision` must be `None` or one of: {choices}.")
        self.max_steps = max_steps
        if isinstance(precision, str):
            # x64 is process-wide and must be enabled before the model and training state are created.
            configure_precision(precision)
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
        self.log_every_n_steps = log_every_n_steps
        resolved_callbacks = tuple(callbacks)
        progress_callbacks = tuple(callback for callback in resolved_callbacks if isinstance(callback, ProgressBar))
        if len(progress_callbacks) > 1:
            raise ValueError("Only one progress-bar callback may be configured.")
        if enable_progress_bar and not progress_callbacks:
            resolved_callbacks += (TQDMProgressBar(refresh_rate=log_every_n_steps),)
        elif not enable_progress_bar:
            for progress_callback in progress_callbacks:
                progress_callback.disable()
        self.enable_progress_bar = enable_progress_bar
        summary_callbacks = tuple(callback for callback in resolved_callbacks if isinstance(callback, ModelSummary))
        if len(summary_callbacks) > 1:
            raise ValueError("Only one model-summary callback may be configured.")
        if enable_model_summary and not summary_callbacks:
            resolved_callbacks += (ModelSummary(warn_if_unavailable=False),)
        self.enable_model_summary = enable_model_summary
        # As in Lightning, checkpoint callbacks run last so a saved checkpoint includes this step's state from every
        # other callback. Identifiers count positions within one callback class, so they are unaffected.
        self.callbacks = tuple(sorted(resolved_callbacks, key=lambda callback: isinstance(callback, ModelCheckpoint)))
        self.should_stop = False
        self._callback_identifiers = _callback_identifiers(self.callbacks)
        checkpoint_callbacks = tuple(callback for callback in self.callbacks if isinstance(callback, ModelCheckpoint))
        if len(checkpoint_callbacks) > 1:
            raise ValueError("Only one `ModelCheckpoint` callback may be configured.")
        self.checkpoint_callback = checkpoint_callbacks[0] if checkpoint_callbacks else None
        prediction_writers = tuple(callback for callback in self.callbacks if isinstance(callback, PredictionWriter))
        if len(prediction_writers) > 1:
            raise ValueError("Only one `PredictionWriter` callback may be configured.")
        self.prediction_writer = prediction_writers[0] if prediction_writers else None
        if loggers is not None:
            if logger is not True:
                raise ValueError("Configure either `logger` or the deprecated `loggers` alias, not both.")
            logger = tuple(loggers)
        self.default_root_dir = Path(default_root_dir).expanduser().resolve()
        self._logger_connector = _LoggerConnector(
            logger,
            self.default_root_dir,
            callbacks=self.callbacks,
            is_global_zero=self.strategy.is_global_zero,
        )
        self._signal_connector = _SignalConnector()
        self._fit_loop = _FitLoop(self)
        self._prediction_loop = _PredictionLoop(self)
        self._data_stage: DataStage | None = None
        for callback in self.callbacks:
            callback.connect(self)
        self.print_environment_info()

    @property
    def logger(self) -> LoggerCollection:
        """Return the configured experiment logger collection.

        Returns:
            Active logger collection.
        """
        return self._logger_connector.logger

    @property
    def interrupted(self) -> bool:
        """Return whether the latest fit was interrupted.

        Returns:
            Current signal connector interruption state.
        """
        return self._signal_connector.interrupted

    @interrupted.setter
    def interrupted(self, value: bool) -> None:
        """Update interruption state for integrations and tests.

        Args:
            value: New interruption flag.
        """
        self._signal_connector.interrupted = value

    @property
    def received_sigterm(self) -> bool:
        """Return whether the latest fit received `SIGTERM`.

        Returns:
            Current signal connector termination flag.
        """
        return self._signal_connector.received_sigterm

    @received_sigterm.setter
    def received_sigterm(self, value: bool) -> None:
        """Update termination state for integrations and tests.

        Args:
            value: New termination flag.
        """
        self._signal_connector.received_sigterm = value

    @property
    def logged_metrics(self) -> Mapping[str, Any]:
        """Return metrics selected for experiment loggers at the latest completed step.

        Returns:
            Immutable-by-convention logger metric mapping.
        """
        return self._logger_connector.logged_metrics

    @property
    def progress_bar_metrics(self) -> Mapping[str, Any]:
        """Return metrics selected for progress display at the latest completed step.

        Returns:
            Immutable-by-convention progress metric mapping.
        """
        return self._logger_connector.progress_bar_metrics

    @property
    def callback_metrics(self) -> Mapping[str, Any]:
        """Return the complete latest metric mapping used by callbacks and fit results.

        Returns:
            Complete merged metric mapping.
        """
        return self._logger_connector.callback_metrics

    def callback_state_dict(self) -> dict[str, Mapping[str, Any]]:
        """Collect JSON-compatible persistent state from every callback.

        Returns:
            Stable callback identifiers mapped to persistent callback state.

        Raises:
            TypeError: If a callback returns a non-mapping or non-JSON-compatible state.
        """
        states: dict[str, Mapping[str, Any]] = {}
        for identifier, callback in zip(self._callback_identifiers, self.callbacks, strict=True):
            state = callback.state_dict()
            if not isinstance(state, Mapping):
                raise TypeError(f"{type(callback).__name__}.state_dict() must return a mapping.")
            try:
                json.dumps(state)
            except (TypeError, ValueError) as error:
                raise TypeError(f"Callback state `{identifier}` must be JSON-compatible.") from error
            states[identifier] = dict(state)
        return states

    def load_callback_state_dict(self, states: Mapping[str, Mapping[str, Any]]) -> None:
        """Restore callback states using stable identifiers, following Lightning.

        Callbacks without a stored state, such as callbacks added since the checkpoint was written or stateless
        callbacks, keep their fresh state. Non-empty stored states without a matching callback are skipped with a
        warning.

        Args:
            states: Checkpoint callback states.

        Raises:
            ValueError: If a callback rejects its stored state.
        """
        identifiers = set(self._callback_identifiers)
        # Empty states come from stateless callbacks, such as progress bars, so removing them loses nothing.
        unmatched = sorted(
            identifier for identifier, state in states.items() if state and identifier not in identifiers
        )
        if unmatched and self.strategy.is_global_zero:
            warnings.warn(
                "The checkpoint stores state for callbacks this Trainer does not have, so that state is not restored: "
                f"{unmatched}. Add these callbacks to the Trainer to resume their state.",
                UserWarning,
                stacklevel=2,
            )
        for identifier, callback in zip(self._callback_identifiers, self.callbacks, strict=True):
            state = states.get(identifier)
            if state:
                callback.load_state_dict(state)

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

    def prepare_batch_source[SourceT](self, source: SourceT) -> SourceT:
        """Prepare persistent sampler state for this Trainer's process-local root device.

        Sources without a callable `prepare` method are returned unchanged, preserving support for ordinary callables
        and finite iterables. Built-in PhiJAX sources transfer finite candidate pools, generation bounds, templates,
        and PRNG keys once instead of transferring sampled training batches from the host on every step.

        Batch sizes are per device. A source exposing `for_devices(count)` is first scaled by this process's device
        count; other sources are used as they are, so their batches are the rows of this process.

        Args:
            source: Training batch callable or iterable, optionally exposing `prepare(device)`.

        Returns:
            Source prepared for the selected Strategy, or the original source when preparation is unsupported.
        """
        source = _for_devices(source, self.strategy.local_device_count)
        prepare = getattr(source, "prepare", None)
        if not callable(prepare):
            return source
        return cast(SourceT, prepare(self.strategy.root_device))

    def prepare_batch(self, batch: Any, replicated: Collection[str] = ()) -> Any:
        """Cast and place one batch according to the configured precision and Strategy.

        NumPy floating leaves are cast on the host before transfer. Device-resident sampler outputs retain their
        placement when the Strategy target is unchanged, while data-parallel Strategies apply final batch sharding.

        Args:
            batch: Arbitrary host or device batch PyTree.
            replicated: Top-level batch names placed in full on every device, such as batches drawn with `all`.

        Returns:
            Precision-normalized batch placed or sharded for compiled execution.
        """
        batch = self.precision.cast_batch(batch)
        # Only pass `replicated` when needed, so custom strategies with the original signature keep working.
        return self.strategy.place_batch(batch, replicated) if replicated else self.strategy.place_batch(batch)

    def fit(
        self,
        module: BasePhiModule,
        *,
        datamodule: PhiDataModule,
        optimizer: optax.GradientTransformation,
        seed: int | jax.Array,
        balancer: LossBalancer | None = None,
        hyperparameters: Mapping[str, Any] | None = None,
        ckpt_path: str | Path | None = None,
        ckpt_step: int | None = None,
        weights_only: bool = False,
    ) -> FitResult:
        """Build a module's model, optionally restore a checkpoint, and fit it.

        Args:
            module: Module blueprint. The Trainer builds its model with
                :meth:`~phijax.core.BasePhiModule.configure_model`, so the blueprint itself is not modified.
            datamodule: Application DataModule supplying training batches and normalization statistics. Its statistics
                apply to fresh and weights-only runs; a full resume uses the statistics stored in the checkpoint.
            optimizer: Optax transformation used to initialize state and compile updates.
            seed: Integer seed or valid JAX key. PhiJAX does not modify Python or NumPy global random state.
            balancer: Functional loss balancer, or `None` for equal static weights.
            hyperparameters: Optional resolved configuration recorded before the first update.
            ckpt_path: Optional Orbax checkpoint root, or `"last"` for the configured callback's latest checkpoint.
            ckpt_step: Exact checkpoint step, or `None` for the latest committed step.
            weights_only: Whether to restore only model weights into the fresh state.

        Returns:
            Bound module, final state, metrics, and termination status.

        Raises:
            TypeError: If `seed` is not an integer or JAX key, or `balancer` does not implement `LossBalancer`.
            ValueError: If the balancer's loss names differ from the module's, the module declares no batch keys, or
                checkpoint options are inconsistent.
        """
        with self._datamodule_stage(datamodule, "fit"):
            root_key = _normalize_seed(seed)
            model_key, runtime_key, sampling_key, balancer_key = jax.random.split(root_key, 4)
            statistics = datamodule.input_statistics()
            input_mean, input_std = (None, None) if statistics is None else statistics
            with jax.default_device(self.strategy.root_device):
                bound_module, model_state = module.configure_model(
                    key=model_key,
                    input_mean=input_mean,
                    input_std=input_std,
                    precision=self.precision,
                    input_periods=datamodule.input_periods(),
                )
                resolved_balancer = StaticLossBalancer(bound_module.loss_names) if balancer is None else balancer
                if not isinstance(resolved_balancer, LossBalancer):
                    raise TypeError("`balancer` must implement `LossBalancer`.")
                if tuple(resolved_balancer.loss_names) != tuple(bound_module.loss_names):
                    raise ValueError("Balancer loss names must exactly match the module loss-name ordering.")
                batch_keys = tuple(bound_module.batch_keys)
                if not batch_keys:
                    raise ValueError("`Trainer.fit()` requires the module to expose at least one batch key.")
                state = initialize_train_state(
                    model_state,
                    optimizer,
                    resolved_balancer.initialize(),
                    runtime_key,
                    self.precision,
                    sampling_key=sampling_key,
                    balancer_key=balancer_key,
                    input_statistics=bound_module.input_statistics,
                    input_periods=bound_module.input_periods,
                )
            state, callback_states = self._restore_state(
                state, ckpt_path, ckpt_step, weights_only=weights_only, restore_callbacks=True
            )
            with jax.default_device(self.strategy.root_device):
                # A full resume restores the checkpoint's statistics and periods. Rebind before compiling, because the
                # train step and balancer update close over the module's model application.
                bound_module = bound_module.bind_inputs(state.input_statistics, state.input_periods)
                train_step = make_train_step(bound_module, resolved_balancer, optimizer, self.precision)
                balancer_update = (
                    resolved_balancer.build_update_plan(bound_module, batch_keys)
                    if isinstance(resolved_balancer, AdaptiveBalancer)
                    else None
                )
            # Record what the model applies, so configured period overrides win over restored periods.
            state = replace(state, input_periods=bound_module.input_periods)
            return self._run_fit(
                bound_module,
                train_step,
                state,
                datamodule=datamodule,
                batch_keys=batch_keys,
                balancer_update=balancer_update,
                hyperparameters=hyperparameters,
                callback_states=callback_states,
            )

    def _run_fit(
        self,
        module: BasePhiModule,
        train_step: TrainStep,
        state: TrainState,
        batches: BatchSource | None = None,
        *,
        datamodule: PhiDataModule | None = None,
        batch_keys: tuple[str, ...] = (),
        balancer_update: BalancerUpdatePlan | None = None,
        hyperparameters: Mapping[str, Any] | None = None,
        callback_states: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> FitResult:
        """Place a built state, resolve the training source, and run the fit loop.

        :meth:`fit` calls this after building, restoring, and compiling. A `datamodule` is set up for the fit stage when
        it supplies the source, and torn down after the loop.

        Args:
            module: Bound module whose model applies `state.input_statistics`.
            train_step: Compiled update that increments `TrainState.step` exactly once.
            state: Fresh or restored functional training state.
            batches: Optional explicit step-indexed callable or finite iterable source.
            datamodule: Optional DataModule supplying the source when `batches` is omitted.
            batch_keys: Objective batch keys requested from `datamodule`.
            balancer_update: Optional host-scheduled adaptive-balancer update.
            hyperparameters: Optional resolved configuration recorded before the first update.
            callback_states: Restored callback states loaded after callback setup, or `None`.

        Returns:
            Bound module, final state, metrics, and termination status.

        Raises:
            ValueError: If neither `batches` nor a `datamodule` with `batch_keys` supplies a source.
            RuntimeError: If a custom training step does not increment `TrainState.step` exactly once.
        """
        with ExitStack() as stack:
            state = self.strategy.place_state(state)
            initial_step = _state_step(state)
            if batches is None:
                if datamodule is None or not batch_keys:
                    raise ValueError("Training requires explicit `batches`, or a `datamodule` with `batch_keys`.")
                if state.sampling_key is None:
                    raise ValueError("A DataModule-owned training source requires `TrainState.sampling_key`.")
                stack.enter_context(self._datamodule_stage(datamodule, "fit"))
                batches = datamodule.train_batch_source(batch_keys, self.strategy.process_key(state.sampling_key))
            batches = self.prepare_batch_source(batches)
            draw_update_batches = (
                None
                if balancer_update is None
                else self._diagnostic_batches(balancer_update, batches, self.strategy.process_key(state.balancer_key))
            )
            # Batches drawn with `all` hold the whole pool, so every device receives them in full.
            full_pools = _full_pool_names(getattr(batches, "batch_sizes", None))
            matmul = self.matmul_precision
            with nullcontext() if matmul is None else jax.default_matmul_precision(matmul):
                return self._fit_loop.run(
                    module,
                    train_step,
                    state,
                    batches,
                    initial_step=initial_step,
                    balancer_update=balancer_update,
                    draw_update_batches=draw_update_batches,
                    hyperparameters=hyperparameters,
                    replicated_batches=full_pools,
                    callback_states=callback_states,
                )

    @contextmanager
    def _datamodule_stage(self, datamodule: PhiDataModule, stage: DataStage) -> Generator[None, None, None]:
        """Run one DataModule stage: prepare data, set up, and always tear down.

        Like Lightning's data connector, the Trainer owns this bookkeeping. Global process zero runs `prepare_data()`,
        and every process waits for it before `setup(stage)`. Entering the stage that is already active does nothing,
        so `fit()` can set up its DataModule once for input statistics and the training source.

        Args:
            datamodule: DataModule to set up.
            stage: Requested `fit` or `predict` stage.

        Yields:
            Control while the stage is set up.

        Raises:
            RuntimeError: If a different DataModule stage is still active.
        """
        if self._data_stage == stage:
            yield
            return
        if self._data_stage is not None:
            raise RuntimeError(f"DataModule stage `{self._data_stage}` is still active; it must end before `{stage}`.")
        self._data_stage = stage
        try:
            if self.strategy.is_global_zero:
                datamodule.prepare_data()
            self.strategy.barrier(f"phijax_prepare_data_{stage}")
            datamodule.setup(stage)
            yield
        finally:
            try:
                datamodule.teardown(stage)
            finally:
                self._data_stage = None

    def _diagnostic_batches(
        self,
        update_plan: BalancerUpdatePlan,
        batches: BatchSource,
        balancer_key: jax.Array,
    ) -> Callable[[int], Any] | None:
        """Build the source of adaptive-balancer diagnostic batches.

        Args:
            update_plan: Balancer update whose `batch_sizes` and `resample` select the diagnostic batches.
            batches: Prepared training source used to sample diagnostic batches.
            balancer_key: Persistent adaptive-balancer diagnostic key from the restored state.

        Returns:
            Function returning the placed diagnostic batches for a global step, or `None` when the update reuses the
            current training batch. Without `resample`, the function returns batches drawn once for this fit.

        Raises:
            TypeError: If diagnostic batches are requested from a source without explicit sampling support.
        """
        batch_sizes = update_plan.batch_sizes
        if batch_sizes is None:
            return None
        full_pools = _full_pool_names(batch_sizes)
        # Diagnostic sizes are per device, like training batch sizes; `all` keeps the whole pool.
        devices = self.strategy.local_device_count
        local_sizes = {name: size if size == "all" else size * devices for name, size in batch_sizes.items()}
        sample = getattr(batches, "sample", None)
        if not callable(sample):
            raise TypeError("Adaptive-balancer diagnostic batches require a training source with `sample()`.")

        def draw(step: int) -> Any:
            """Draw diagnostic batches for one global step.

            Args:
                step: Global optimizer step folded into the persistent balancer key.

            Returns:
                Diagnostic batches placed on the Trainer's device.
            """
            # Keys depend only on `balancer_key` and the global step, so resumed fits draw the same batches.
            return self.prepare_batch(sample(jax.random.fold_in(balancer_key, step), local_sizes), full_pools)

        if update_plan.resample:
            return draw
        fixed = draw(0)
        return lambda step: fixed

    def predict(
        self,
        model: BasePhiModule | FitResult,
        batches: Iterable[Mapping[str, Any]] | None = None,
        *,
        datamodule: PhiDataModule | None = None,
        ckpt_path: str | Path | None = None,
        ckpt_step: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        return_predictions: bool = True,
    ) -> np.ndarray | None:
        """Predict with a fit result, or with a module blueprint loaded from a checkpoint.

        This follows Lightning's `Trainer.predict(model, ckpt_path=...)`. A :class:`FitResult` predicts with its bound
        module and final model state, or with the model state stored at `ckpt_path`. An unbound :class:`PhiModule`
        requires `ckpt_path`: the Trainer builds its model with the checkpoint's input statistics and periods and
        restores only the model state, so prediction needs no optimizer, balancer, or fit-stage data.

        Checkpoints written by PhiJAX 0.2.0b5 and earlier store no input statistics or periods. With one, a fit result
        keeps its module's values, and a module takes them from the fit stage of `datamodule` with a warning.

        When `batches` is omitted, the Trainer sets up `datamodule` for prediction and requests its batch source.
        Prediction is skipped when neither source is available.

        Args:
            model: Fit result, or unbound module blueprint restored from `ckpt_path`.
            batches: Optional finite prediction batch iterable. When omitted, use `datamodule`.
            datamodule: Optional DataModule supplying prediction batches, and the input statistics and periods of a
                legacy checkpoint.
            ckpt_path: Optional Orbax checkpoint root, or `"last"` for the configured callback's latest checkpoint.
            ckpt_step: Exact checkpoint step, or `None` for the latest committed step.
            metadata: Optional immutable metadata exposed to prediction callbacks.
            return_predictions: Whether to concatenate and return host predictions.

        Returns:
            Concatenated host predictions, or `None` when collection is disabled or prediction data is unavailable.

        Raises:
            ValueError: If a module blueprint has no `ckpt_path`, checkpoint options are invalid, a legacy checkpoint
                has no `datamodule` to take inputs from, or a provided batch source is empty.
            BaseException: Re-raises prediction lifecycle errors after cleanup.
        """
        if ckpt_path is not None:
            module, model_state = self._load_prediction_model(model, ckpt_path, ckpt_step, datamodule)
        elif isinstance(model, FitResult):
            module, model_state = model.module, model.state.model_state
        else:
            raise ValueError(
                "Predicting with a module blueprint requires `ckpt_path`. Pass the `FitResult` returned by `fit()` to "
                "predict with in-memory state."
            )
        with ExitStack() as stack:
            if batches is None:
                if datamodule is None:
                    return None
                stack.enter_context(self._datamodule_stage(datamodule, "predict"))
                batches = datamodule.predict_batch_source()
                if batches is None:
                    return None
            # Every process works through the same chunks, so a per-device chunk spans all devices.
            batches = _for_devices(batches, len(self.strategy.devices))
            matmul = self.matmul_precision
            with nullcontext() if matmul is None else jax.default_matmul_precision(matmul):
                return self._prediction_loop.run(
                    module,
                    model_state,
                    batches,
                    metadata=metadata,
                    return_predictions=return_predictions,
                )

    def _load_prediction_model(
        self,
        model: BasePhiModule | FitResult,
        ckpt_path: str | Path,
        ckpt_step: int | None,
        datamodule: PhiDataModule | None,
    ) -> tuple[BasePhiModule, Any]:
        """Bind the prediction module to a checkpoint's inputs and restore only its model state.

        Args:
            model: Fit result whose model state is the restore template, or unbound module blueprint.
            ckpt_path: Orbax checkpoint root, or `"last"` for the configured callback's latest checkpoint.
            ckpt_step: Exact checkpoint step, or `None` for the latest committed step.
            datamodule: Optional DataModule whose fit stage supplies the inputs of a legacy checkpoint.

        Returns:
            Bound module and restored model state.

        Raises:
            ValueError: If a legacy checkpoint is restored into a module blueprint without a `datamodule`.
        """
        with self._checkpoint_backend(ckpt_path, ckpt_step) as (checkpoint_io, step):
            try:
                statistics = checkpoint_io.restore_input_statistics(step)
                periods = checkpoint_io.restore_input_periods(step)
            except LegacyCheckpointError:
                if isinstance(model, FitResult):
                    statistics, periods = model.module.input_statistics, model.module.input_periods
                elif datamodule is None:
                    raise ValueError(
                        f"Checkpoint step {step} was written before PhiJAX stored input statistics and periods. Pass "
                        "the training `datamodule` so prediction can take them from its fit stage."
                    ) from None
                else:
                    warnings.warn(
                        f"Checkpoint step {step} was written before PhiJAX stored input statistics and periods. "
                        "Prediction takes them from the fit stage of `datamodule`.",
                        UserWarning,
                        stacklevel=3,
                    )
                    with self._datamodule_stage(datamodule, "fit"):
                        statistics, periods = datamodule.input_statistics(), datamodule.input_periods()
            with jax.default_device(self.strategy.root_device):
                if isinstance(model, FitResult):
                    module = model.module.bind_inputs(as_input_statistics(statistics), periods)
                    template = model.state.model_state
                else:
                    input_mean, input_std = (None, None) if statistics is None else statistics
                    # The key only shapes the template; every model-state leaf is restored from the checkpoint.
                    module, template = model.configure_model(
                        key=jax.random.key(0),
                        input_mean=input_mean,
                        input_std=input_std,
                        precision=self.precision,
                        input_periods=periods,
                    )
                    template = self.precision.cast_model_state(template)
            return module, checkpoint_io.restore_model_state(template, step)

    @contextmanager
    def _checkpoint_backend(
        self,
        ckpt_path: str | Path,
        step: int | None,
        *,
        allow_missing: bool = False,
    ) -> Generator[tuple[CheckpointIO, int | None], None, None]:
        """Open the backend of an explicit checkpoint root or of the configured callback, and close it afterwards.

        Args:
            ckpt_path: Explicit checkpoint root, or `"last"` for the configured `ModelCheckpoint` backend.
            step: Exact checkpoint step, or `None` for the latest committed step.
            allow_missing: Whether a backend without committed checkpoints yields `None` instead of raising.

        Yields:
            Open backend and the resolved checkpoint step, or `None` when `allow_missing` finds no checkpoint.

        Raises:
            ValueError: If `ckpt_path` is empty, `step` is invalid, or `"last"` has no configured `ModelCheckpoint`.
            FileNotFoundError: If the backend contains no committed checkpoint and `allow_missing` is false.
        """
        if ckpt_path == "last":
            if self.checkpoint_callback is None:
                raise ValueError("`ckpt_path='last'` requires a configured `ModelCheckpoint` callback.")
            checkpoint_io = self.checkpoint_callback.checkpoint_io
        else:
            if isinstance(ckpt_path, str) and not ckpt_path.strip():
                raise ValueError("`ckpt_path` must be a non-empty checkpoint-root path or `None`.")
            if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
                raise ValueError("`ckpt_step` must be a nonnegative integer or `None`.")
            checkpoint_io = OrbaxCheckpointIO(ckpt_path, enable_async_checkpointing=False)
        try:
            checkpoint_io.open()
            resolved_step = checkpoint_io.latest_step if step is None else step
            if resolved_step is None and not allow_missing:
                raise FileNotFoundError(f"No committed checkpoint exists below `{ckpt_path}`.")
            yield checkpoint_io, resolved_step
        finally:
            checkpoint_io.close()

    def _restore_state(
        self,
        state: TrainState,
        ckpt_path: str | Path | None,
        ckpt_step: int | None,
        *,
        weights_only: bool,
        restore_callbacks: bool = False,
    ) -> tuple[TrainState, Mapping[str, Mapping[str, Any]] | None]:
        """Restore an explicit checkpoint path or the configured callback's latest state.

        As in Lightning, `"last"` without a committed checkpoint warns and returns `state`, so one configuration can
        start a run and also resume it after preemption.

        Args:
            state: Fresh state or restore template.
            ckpt_path: Explicit checkpoint root, `"last"`, or `None`.
            ckpt_step: Optional explicit checkpoint step.
            weights_only: Whether to replace only `model_state`.
            restore_callbacks: Whether a full restore also returns the stored callback states.

        Returns:
            Restored state, or `state` when no checkpoint is selected, and the callback states to load before
            fit-start hooks, or `None`.

        Raises:
            ValueError: If restore options are inconsistent or checkpointing is unavailable.
            FileNotFoundError: If an explicit checkpoint root has no committed checkpoint.
        """
        if ckpt_path is None:
            if weights_only or ckpt_step is not None:
                raise ValueError("`weights_only` and `ckpt_step` require a non-null `ckpt_path`.")
            return state, None
        if ckpt_path == "last" and ckpt_step is not None:
            raise ValueError("`ckpt_path='last'` cannot be combined with an explicit `ckpt_step`.")
        with self._checkpoint_backend(ckpt_path, ckpt_step, allow_missing=ckpt_path == "last") as (checkpoint_io, step):
            if step is None:
                if self.strategy.is_global_zero:
                    warnings.warn(
                        "`ckpt_path='last'` is set, but no checkpoint has been saved yet, so `fit()` starts from fresh "
                        "state.",
                        UserWarning,
                        stacklevel=3,
                    )
                return state, None
            if weights_only:
                return replace(state, model_state=checkpoint_io.restore_model_state(state.model_state, step)), None
            restored = checkpoint_io.restore(state, step)
            return restored, checkpoint_io.restore_callback_states(step) if restore_callbacks else None

    def close(self) -> None:
        """Release any checkpoint resources opened outside a Trainer task.

        Fit and prediction stages already close their resources automatically. Calling this method is optional and
        remains supported for explicit lifecycle management and context-manager use.
        """
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


def _full_pool_names(batch_sizes: Mapping[str, Any] | None) -> frozenset[str]:
    """Return the batch names drawn with the `all` policy.

    Args:
        batch_sizes: Optional mapping from batch name to an integer size or `all`.

    Returns:
        Names whose batches hold a whole pool.
    """
    return frozenset(name for name, size in (batch_sizes or {}).items() if size == "all")


def _for_devices[SourceT](source: SourceT, count: int) -> SourceT:
    """Scale a batch source's per-device sizes to the devices that share each batch.

    Args:
        source: Batch source, optionally exposing `for_devices(count)` like the built-in PhiJAX sources.
        count: Number of devices that split each batch.

    Returns:
        The scaled source, or `source` unchanged when it has no `for_devices` method.
    """
    for_devices = getattr(source, "for_devices", None)
    return cast(SourceT, for_devices(count)) if callable(for_devices) else source


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


def _normalize_seed(seed: int | jax.Array) -> jax.Array:
    """Convert an integer or validate one unbatched JAX PRNG key.

    Args:
        seed: Python integer or typed or legacy JAX key.

    Returns:
        Unbatched JAX key without modifying any global random state.

    Raises:
        TypeError: If `seed` is Boolean, non-integral, or not a valid unbatched JAX key.
    """
    if isinstance(seed, int) and not isinstance(seed, bool):
        return jax.random.key(seed)
    if not isinstance(seed, jax.Array):
        raise TypeError("`seed` must be an integer or an unbatched JAX PRNG key.")
    try:
        key_data = jax.random.key_data(seed)
    except (TypeError, ValueError) as error:
        raise TypeError("`seed` must be an integer or an unbatched JAX PRNG key.") from error
    if key_data.shape != (2,):
        raise TypeError("`seed` must be an unbatched JAX PRNG key.")
    return seed


def _callback_identifiers(callbacks: Sequence[Callback]) -> tuple[str, ...]:
    """Create stable identifiers including an occurrence index for repeated callback types.

    Args:
        callbacks: Ordered Trainer callback sequence.

    Returns:
        Stable fully qualified callback identifiers.
    """
    counts: dict[str, int] = {}
    identifiers = []
    for callback in callbacks:
        base = f"{type(callback).__module__}.{type(callback).__qualname__}"
        index = counts.get(base, 0)
        counts[base] = index + 1
        identifiers.append(f"{base}:{index}")
    return tuple(identifiers)


__all__ = ["BatchSource", "FitResult", "TrainStep", "Trainer"]
