import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Self, cast

import jax
import numpy as np
import optax

from phijax.balancers import LossBalancer, StaticLossBalancer
from phijax.callbacks import (
    Callback,
    ModelCheckpoint,
    ModelSummary,
    PredictionWriter,
    ProgressBar,
    TQDMProgressBar,
)
from phijax.core import BasePhiModule, PhiModule
from phijax.data import PhiDataModule
from phijax.training.assembly import build_training_plan
from phijax.training.checkpointing import OrbaxCheckpointIO, restore_checkpoint
from phijax.training.connectors.logger_connector import _LoggerConnector
from phijax.training.connectors.signal_connector import _SignalConnector
from phijax.training.loggers import ExperimentLogger, LoggerCollection
from phijax.training.loops.fit_loop import _BalancerUpdateRuntime, _FitLoop, _state_step
from phijax.training.loops.prediction_loop import _PredictionLoop
from phijax.training.plans import TrainingPlan
from phijax.training.precision import MatmulPrecision, PrecisionName, PrecisionPolicy
from phijax.training.results import FitResult
from phijax.training.state import TrainState, initialize_train_state
from phijax.training.steps import TrainStep, make_train_step
from phijax.training.strategies import Accelerator, DeviceSelection, Strategy, create_strategy
from phijax.types import JaxDevice

type BatchSource = Iterable[Any] | Callable[[int], Any]

_MATMUL_PRECISIONS = frozenset({"default", "high", "highest"})


class Trainer:
    """Coordinate compiled JAX updates, callbacks, logging, and checkpointing on the host.

    The trainer never owns a mutable model object and never places Python hooks inside :func:`jax.jit`. Numerical
    behavior remains in the configured training plan or supplied compiled step; this class orchestrates its lifecycle.

    Attributes:
        prediction_writer: Configured prediction artifact callback, or `None`.
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
        initial_loss_scale: float = 32768.0,
        loss_scale_growth_interval: int = 2000,
        deterministic: bool = True,
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
            max_steps: Maximum number of batches processed by one fit call.
            accelerator: Device backend selected when `strategy` is absent. Choose `auto`, `cpu`, `gpu`, or `tpu`.
            devices: Positive device count, explicit device indices, or `auto` for all matching visible devices.
            precision: Precision name or resolved policy. Supported names are `64-true`, `32-true`, `16-true`,
                `bf16-true`, `16-mixed`, and `bf16-mixed`; common dtype aliases are also accepted.
            matmul_precision: Optional `default`, `high`, or `highest` JAX matrix-multiplication precision override.
            derivative_dtype: Optional floating batch and coordinate-derivative dtype override.
            initial_loss_scale: Initial dynamic scale for `16-mixed` training.
            loss_scale_growth_interval: Finite optimizer updates between FP16 scale increases.
            deterministic: Whether the caller promises deterministic data and step behavior.
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
            ValueError: If a step, interval, callback, logger, or matrix-multiplication precision option is invalid.
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
        resolved_callbacks = tuple(callbacks)
        progress_callbacks = tuple(callback for callback in resolved_callbacks if isinstance(callback, ProgressBar))
        if len(progress_callbacks) > 1:
            raise ValueError("Only one progress-bar callback may be configured.")
        if enable_progress_bar and not progress_callbacks:
            resolved_callbacks += (TQDMProgressBar(total=max_steps, refresh_rate=log_every_n_steps),)
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
        self.callbacks = resolved_callbacks
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
        self._pending_callback_states: Mapping[str, Mapping[str, Any]] | None = None
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
        """Restore callback states using stable identifiers.

        Args:
            states: Checkpoint callback states.

        Raises:
            ValueError: If checkpoint and Trainer callback identifiers differ.
        """
        expected = set(self._callback_identifiers)
        received = set(states)
        if received != expected:
            missing = sorted(expected - received)
            unexpected = sorted(received - expected)
            raise ValueError(f"Checkpoint callback state is incompatible; missing={missing}, unexpected={unexpected}.")
        for identifier, callback in zip(self._callback_identifiers, self.callbacks, strict=True):
            callback.load_state_dict(states[identifier])

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
        key: jax.Array,
        *,
        sampling_key: jax.Array | None = None,
        balancer_key: jax.Array | None = None,
    ) -> TrainState:
        """Initialize functional state with this trainer's precision policy.

        Args:
            model_state: Explicit differentiable model parameter PyTree.
            optimizer: Optax transformation used to initialize optimizer slots.
            balancer_state: Initial loss-balancer state.
            key: Root training key, or model-runtime key when both persistent keys are supplied.
            sampling_key: Optional explicit DataModule sampling key.
            balancer_key: Optional explicit adaptive-balancer diagnostic key.

        Returns:
            Complete precision-aware training state.
        """
        return initialize_train_state(
            model_state,
            optimizer,
            balancer_state,
            key,
            self.precision,
            sampling_key=sampling_key,
            balancer_key=balancer_key,
        )

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
        module: PhiModule,
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
        """Initialize and fit a module blueprint through the concise application API.

        Args:
            module: Unbound :class:`PhiModule` containing a model factory and objective.
            datamodule: Application DataModule supplying normalization statistics and training batches.
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
            TypeError: If `seed` is not an integer or JAX key.
            ValueError: If model, balancing, checkpoint, or DataModule options are inconsistent.
        """
        datamodule.prepare_stage("fit")
        fit_state_owns_teardown = False
        try:
            root_key = _normalize_seed(seed)
            model_key, runtime_key, sampling_key, balancer_key = jax.random.split(root_key, 4)
            statistics = datamodule.input_statistics()
            input_mean, input_std = (None, None) if statistics is None else statistics
            with jax.default_device(self.strategy.root_device):
                bound_module, model_state = module.prepare_model(
                    key=model_key,
                    input_mean=input_mean,
                    input_std=input_std,
                    precision=self.precision,
                )
                resolved_balancer = StaticLossBalancer(bound_module.loss_names) if balancer is None else balancer
                plan = build_training_plan(self, bound_module, resolved_balancer, optimizer)
                state = self.initialize_state(
                    model_state,
                    optimizer,
                    resolved_balancer.initialize(),
                    runtime_key,
                    sampling_key=sampling_key,
                    balancer_key=balancer_key,
                )
            fit_state_owns_teardown = True
            return self.fit_state(
                bound_module,
                plan,
                state,
                hyperparameters=hyperparameters,
                ckpt_path=ckpt_path,
                ckpt_step=ckpt_step,
                weights_only=weights_only,
                datamodule=datamodule,
            )
        finally:
            if not fit_state_owns_teardown:
                datamodule.teardown_stage("fit")

    def fit_state(
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
    ) -> FitResult:
        """Fit an explicitly initialized module, state, and resolved numerical plan.

        Args:
            module: Bound custom module owning model, objective, and lifecycle behavior.
            training: Resolved training plan, or a compiled step when `batches` is supplied explicitly.
            state: Initial functional training state and restore template.
            batches: Optional explicit step-indexed callable or finite iterable source.
            hyperparameters: Optional resolved configuration recorded before the first update.
            ckpt_path: Optional Orbax checkpoint root, or `"last"` for the configured callback's latest checkpoint.
            ckpt_step: Exact checkpoint step, or `None` for the latest committed step.
            weights_only: Whether to restore only model weights into the fresh state.
            datamodule: Optional DataModule supplying the source when `batches` is omitted.

        Returns:
            Bound module, final state, metrics, and termination status.

        Raises:
            ValueError: If checkpoint restoration or DataModule source options are incomplete or invalid.
            RuntimeError: If a custom training step does not increment `TrainState.step` exactly once.
            BaseException: Re-raises training lifecycle errors after cleanup.
        """
        try:
            plan = training if isinstance(training, TrainingPlan) else TrainingPlan(training)
            self._pending_callback_states = None
            state = self._restore_state(
                state,
                ckpt_path,
                ckpt_step,
                weights_only=weights_only,
                restore_callbacks=not weights_only and ckpt_path is not None,
            )
            state = self.strategy.place_state(state)
            initial_step = _state_step(state)
            batches = self._training_batch_source(plan, batches, datamodule, state.sampling_key)
            balancer_update = self._balancer_update_runtime(plan, batches, state.balancer_key, initial_step)
            precision_context = (
                nullcontext() if self.matmul_precision is None else jax.default_matmul_precision(self.matmul_precision)
            )
            with precision_context:
                return self._fit_loop.run(
                    module,
                    plan.train_step,
                    state,
                    batches,
                    initial_step=initial_step,
                    balancer_update=balancer_update,
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
            raise ValueError("`Trainer.fit_state()` requires either explicit `batches` or a `datamodule`.")
        if not plan.batch_keys:
            raise ValueError("A DataModule-owned training source requires non-empty `TrainingPlan.batch_keys`.")
        if sampling_key is None:
            raise ValueError("A DataModule-owned training source requires an explicit `sampling_key`.")
        datamodule.prepare_stage("fit")
        return self.prepare_batch_source(datamodule.train_batch_source(plan.batch_keys, sampling_key))

    def _balancer_update_runtime(
        self,
        plan: TrainingPlan,
        batches: BatchSource,
        balancer_key: jax.Array,
        initial_step: int,
    ) -> _BalancerUpdateRuntime | None:
        """Resolve optional adaptive-balancer diagnostics for host scheduling.

        Args:
            plan: Training plan containing the compiled optimizer step and optional update schedule.
            batches: Prepared training source used for fixed diagnostic sampling when requested.
            balancer_key: Persistent adaptive-balancer diagnostic key from the restored state.
            initial_step: Restored global step folded into fixed diagnostic sampling.

        Returns:
            Resolved adaptive update runtime, or `None` for static balancing.

        Raises:
            TypeError: If fixed diagnostics are requested from a source without explicit sampling support.
        """
        update_plan = plan.balancer_update
        if update_plan is None:
            return None
        update_batches = None
        if update_plan.batch_sizes is not None:
            sample = getattr(batches, "sample", None)
            if not callable(sample):
                raise TypeError("Fixed adaptive-balancer diagnostics require a training source with `sample()`.")
            diagnostic_key = jax.random.fold_in(balancer_key, initial_step)
            update_batches = self.prepare_batch(sample(diagnostic_key, update_plan.batch_sizes))
        return _BalancerUpdateRuntime(
            update=update_plan.update,
            batches=update_batches,
            every_n_steps=update_plan.every_n_steps,
            update_start_step=update_plan.update_start_step,
        )

    def predict(
        self,
        result: FitResult,
        batches: Iterable[Mapping[str, Any]] | None = None,
        *,
        datamodule: PhiDataModule | None = None,
        ckpt_path: str | Path | None = None,
        ckpt_step: int | None = None,
        metadata: Mapping[str, Any] | None = None,
        return_predictions: bool = True,
    ) -> np.ndarray | None:
        """Predict from the bound module and functional state returned by :meth:`fit`.

        Args:
            result: Completed fit result containing the bound module and state.
            batches: Optional finite prediction batch iterable. When omitted, use `datamodule`.
            datamodule: Optional DataModule supplying prediction batches.
            ckpt_path: Optional Orbax checkpoint root, or `"last"` for the configured callback's latest checkpoint.
            ckpt_step: Exact checkpoint step, or `None` for the latest committed step.
            metadata: Optional immutable metadata exposed to prediction callbacks.
            return_predictions: Whether to concatenate and return host predictions.

        Returns:
            Concatenated host predictions, or `None` when collection is disabled or prediction data is unavailable.
        """
        return self.predict_state(
            result.module,
            result.state,
            batches,
            datamodule=datamodule,
            ckpt_path=ckpt_path,
            ckpt_step=ckpt_step,
            metadata=metadata,
            return_predictions=return_predictions,
        )

    def predict_state(
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
        """Predict from an explicitly bound module and functional state template.

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
            restored_state = self._restore_state(state, ckpt_path, ckpt_step, weights_only=ckpt_path is not None)
            precision_context = (
                nullcontext() if self.matmul_precision is None else jax.default_matmul_precision(self.matmul_precision)
            )
            with precision_context:
                return self._prediction_loop.run(
                    module,
                    restored_state.model_state,
                    batches,
                    metadata=metadata,
                    return_predictions=return_predictions,
                )
        finally:
            if datamodule is not None:
                datamodule.teardown_stage("predict")

    def _restore_state(
        self,
        state: TrainState,
        ckpt_path: str | Path | None,
        ckpt_step: int | None,
        *,
        weights_only: bool,
        restore_callbacks: bool = False,
    ) -> TrainState:
        """Restore an explicit checkpoint path or the configured callback's latest state.

        Args:
            state: Fresh state or restore template.
            ckpt_path: Explicit checkpoint root, `"last"`, or `None`.
            ckpt_step: Optional explicit checkpoint step.
            weights_only: Whether to replace only `model_state`.
            restore_callbacks: Whether to stage callback state for restoration before fit-start hooks.

        Returns:
            Restored state, or `state` when no checkpoint is selected.

        Raises:
            ValueError: If `"last"` is combined with an explicit step or checkpointing is unavailable.
            FileNotFoundError: If the configured callback has no committed checkpoint.
        """
        if ckpt_path != "last":
            if not restore_callbacks or ckpt_path is None:
                return restore_checkpoint(state, ckpt_path, weights_only=weights_only, step=ckpt_step)
            with OrbaxCheckpointIO(ckpt_path, max_to_keep=None, enable_async_checkpointing=False) as checkpoint_io:
                restored = checkpoint_io.restore(state, ckpt_step)
                self._pending_callback_states = checkpoint_io.restore_callback_states(ckpt_step)
                return restored
        if ckpt_step is not None:
            raise ValueError("`ckpt_path='last'` cannot be combined with an explicit `ckpt_step`.")
        if self.checkpoint_callback is None:
            raise ValueError("`ckpt_path='last'` requires a configured `ModelCheckpoint` callback.")
        checkpoint_io = self.checkpoint_callback.checkpoint_io
        try:
            checkpoint_io.open()
            latest_step = checkpoint_io.latest_step
            if latest_step is None:
                raise FileNotFoundError("The configured checkpoint backend contains no committed state.")
            if weights_only:
                return checkpoint_io.restore_weights(state, latest_step)
            restored = checkpoint_io.restore(state, latest_step)
            if restore_callbacks:
                self._pending_callback_states = checkpoint_io.restore_callback_states(latest_step)
            return restored
        finally:
            checkpoint_io.close()

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
        checkpoint_io = self.checkpoint_callback.checkpoint_io
        try:
            checkpoint_io.open()
            return checkpoint_io.restore_weights(state, step)
        finally:
            checkpoint_io.close()

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
        try:
            checkpoint_io.open()
            checkpoint_step = checkpoint_io.latest_step
            if checkpoint_step is None:
                raise FileNotFoundError("The configured checkpoint backend contains no committed state.")
            checkpoint_directory = getattr(checkpoint_io, "directory", None)
            if checkpoint_directory is None:
                raise ValueError("The configured checkpoint backend does not expose a filesystem `directory`.")
            return Path(checkpoint_directory).expanduser().resolve(), checkpoint_step
        finally:
            checkpoint_io.close()

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
