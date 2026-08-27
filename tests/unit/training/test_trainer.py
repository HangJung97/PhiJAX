import signal
from collections.abc import Mapping
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

import phijax.training.trainer as trainer_module
from phijax.balancers import BalancerState, StaticLossBalancer
from phijax.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    PredictionContext,
    PredictionWriter,
    TrainerContext,
)
from phijax.data import ChunkedPredictionSource, HostPool, PhiDataModule
from phijax.module import BasePhiModule, PhiModule, PhiModuleContext
from phijax.training import (
    ExperimentLogger,
    SingleDeviceStrategy,
    Strategy,
    Trainer,
    TrainingPlan,
    TrainState,
)
from phijax.training.trainer import _FitSignalHandler


class _RecordingCallback(Callback):
    """Record trainer lifecycle events for ordering assertions."""

    def __init__(self, timeline: list[str] | None = None) -> None:
        """Initialize an empty event list.

        Args:
            timeline: Optional shared callback-and-module event list.
        """
        self.events: list[str] = []
        self.timeline = timeline

    def _record(self, event: str) -> None:
        """Record one local and shared callback event.

        Args:
            event: Unprefixed callback event description.
        """
        self.events.append(event)
        if self.timeline is not None:
            self.timeline.append(f"callback:{event}")

    def setup(self) -> None:
        """Record callback setup."""
        self._record("setup")

    def on_fit_start(self, context: TrainerContext) -> None:
        """Record fit start.

        Args:
            context: Initial trainer context.
        """
        self._record(f"fit_start:{context.step}")

    def on_train_batch_start(self, context: TrainerContext) -> None:
        """Record batch start.

        Args:
            context: Pre-update trainer context.
        """
        self._record(f"batch_start:{context.step}")

    def on_train_batch_end(self, context: TrainerContext) -> bool:
        """Record batch end without requesting a stop.

        Args:
            context: Post-update trainer context.

        Returns:
            Always `False`.
        """
        self._record(f"batch_end:{context.step}")
        return False

    def on_fit_end(self, context: TrainerContext) -> None:
        """Record fit end.

        Args:
            context: Final trainer context.
        """
        self._record(f"fit_end:{context.step}")

    def on_exception(self, exception: BaseException, context: TrainerContext) -> None:
        """Record callback exception handling.

        Args:
            exception: Raised training exception.
            context: Most recent trainer context.
        """
        self._record(f"exception:{type(exception).__name__}:{context.step}")

    def on_predict_start(self, context: PredictionContext) -> None:
        """Record prediction start.

        Args:
            context: Initial prediction context.
        """
        del context
        self._record("predict_start")

    def on_predict_epoch_start(self, context: PredictionContext) -> None:
        """Record prediction-pass start.

        Args:
            context: Initial prediction-pass context.
        """
        del context
        self._record("predict_epoch_start")

    def on_predict_batch_start(self, context: PredictionContext) -> None:
        """Record one starting prediction batch.

        Args:
            context: Prediction batch context.
        """
        self._record(f"predict_batch_start:{context.batch_index}")

    def on_predict_batch_end(self, context: PredictionContext) -> None:
        """Record one completed prediction batch.

        Args:
            context: Prediction batch context.
        """
        self._record(f"predict_batch_end:{context.batch_index}")

    def on_predict_epoch_end(self, context: PredictionContext) -> None:
        """Record prediction-pass completion.

        Args:
            context: Final prediction-pass context.
        """
        del context
        self._record("predict_epoch_end")

    def on_predict_end(self, context: PredictionContext) -> None:
        """Record prediction completion.

        Args:
            context: Final prediction context.
        """
        del context
        self._record("predict_end")

    def teardown(self) -> None:
        """Record callback teardown."""
        self._record("teardown")


class _PredictionContextCallback(Callback):
    """Retain the final prediction context for source-metadata assertions."""

    def __init__(self) -> None:
        """Initialize without a completed prediction context."""
        self.context: PredictionContext | None = None

    def on_predict_end(self, context: PredictionContext) -> None:
        """Retain one final prediction context.

        Args:
            context: Final context dispatched by the trainer.
        """
        self.context = context


class _RecordingModule(BasePhiModule):
    """Record overridable application lifecycle hooks used by the trainer."""

    def __init__(self, timeline: list[str] | None = None) -> None:
        """Initialize the module and its empty event list.

        Args:
            timeline: Optional shared callback-and-module event list.
        """
        super().__init__(name="test")
        self.events: list[str] = []
        self.timeline = timeline

    def _record(self, event: str) -> None:
        """Record one local and shared module event.

        Args:
            event: Unprefixed module event description.
        """
        self.events.append(event)
        if self.timeline is not None:
            self.timeline.append(f"module:{event}")

    @property
    def loss_names(self) -> tuple[str, ...]:
        """Return the synthetic loss name.

        Returns:
            One stable scalar loss name.
        """
        return ("loss",)

    def forward(self, model_state: Any, inputs: jax.Array) -> jax.Array:
        """Apply the synthetic scalar model.

        Args:
            model_state: Scalar model state.
            inputs: Input values.

        Returns:
            Inputs multiplied by the scalar model weight.
        """
        return model_state["weight"] * inputs

    def training_step(
        self,
        model_state: Any,
        batches: dict[str, dict[str, jax.Array]],
    ) -> Mapping[str, jax.Array]:
        """Evaluate the synthetic scalar loss.

        Args:
            model_state: Scalar model state.
            batches: Nested input batches.

        Returns:
            One mean-squared model output loss.
        """
        outputs = self(model_state, batches["data"]["inputs"])
        return {"loss": jnp.mean(outputs**2)}

    def setup(self) -> None:
        """Record module setup."""
        self._record("setup")

    def on_fit_start(self, model_state: Any, context: PhiModuleContext) -> Any:
        """Record fit start.

        Args:
            model_state: Initial model state.
            context: Initial module context.

        Returns:
            Unchanged model state.
        """
        self._record(f"fit_start:{context.step}")
        return model_state

    def on_train_batch_start(
        self,
        model_state: Any,
        batch: Any,
        context: PhiModuleContext,
    ) -> tuple[Any, Any]:
        """Record batch start.

        Args:
            model_state: Current model state.
            batch: Current training batch.
            context: Pre-update module context.

        Returns:
            Unchanged model state and batch.
        """
        self._record(f"batch_start:{context.step}")
        return model_state, batch

    def on_train_batch_end(
        self,
        model_state: Any,
        context: PhiModuleContext,
    ) -> tuple[Any, Mapping[str, Any]]:
        """Record batch end.

        Args:
            model_state: Updated model state.
            context: Post-update module context.

        Returns:
            Unchanged model state and metrics.
        """
        self._record(f"batch_end:{context.step}")
        return model_state, context.metrics

    def on_predict_start(self, model_state: Any, context: PredictionContext) -> None:
        """Record prediction start.

        Args:
            model_state: Explicit prediction state.
            context: Initial prediction context.
        """
        del model_state, context
        self._record("predict_start")

    def on_predict_epoch_start(self, model_state: Any, context: PredictionContext) -> None:
        """Record prediction-pass start.

        Args:
            model_state: Explicit prediction state.
            context: Initial prediction-pass context.
        """
        del model_state, context
        self._record("predict_epoch_start")

    def on_predict_batch_start(self, model_state: Any, context: PredictionContext) -> None:
        """Record one starting prediction batch.

        Args:
            model_state: Explicit prediction state.
            context: Prediction batch context.
        """
        del model_state
        self._record(f"predict_batch_start:{context.batch_index}")

    def on_predict_batch_end(self, model_state: Any, context: PredictionContext) -> None:
        """Record one completed prediction batch.

        Args:
            model_state: Explicit prediction state.
            context: Prediction batch context.
        """
        del model_state
        self._record(f"predict_batch_end:{context.batch_index}")

    def on_predict_epoch_end(self, model_state: Any, context: PredictionContext) -> None:
        """Record prediction-pass completion.

        Args:
            model_state: Explicit prediction state.
            context: Final prediction-pass context.
        """
        del model_state, context
        self._record("predict_epoch_end")

    def on_predict_end(self, model_state: Any, context: PredictionContext) -> None:
        """Record prediction completion.

        Args:
            model_state: Explicit prediction state.
            context: Final prediction context.
        """
        del model_state, context
        self._record("predict_end")

    def on_fit_end(self, model_state: Any, context: PhiModuleContext) -> Any:
        """Record fit end.

        Args:
            model_state: Terminal model state.
            context: Terminal module context.

        Returns:
            Unchanged model state.
        """
        self._record(f"fit_end:{context.step}")
        return model_state

    def on_exception(self, exception: BaseException, context: PhiModuleContext) -> None:
        """Record a trainer exception.

        Args:
            exception: Raised training exception.
            context: Most recent module context.
        """
        self._record(f"exception:{type(exception).__name__}:{context.step}")

    def teardown(self) -> None:
        """Record module teardown."""
        self._record("teardown")


class _TransformingModule(_RecordingModule):
    """Return distinguishable state, batch, and metric replacements from module hooks."""

    def on_fit_start(self, model_state: Any, context: PhiModuleContext) -> Any:
        """Increment model state after the callback fit-start hook.

        Args:
            model_state: Initial model state.
            context: Initial module context.

        Returns:
            Model state with its scalar weight incremented by one.
        """
        self._record(f"fit_start:{context.step}")
        return {"weight": model_state["weight"] + 1.0}

    def on_train_batch_start(
        self,
        model_state: Any,
        batch: Any,
        context: PhiModuleContext,
    ) -> tuple[Any, Any]:
        """Double the batch increment after the callback batch-start hook.

        Args:
            model_state: Current model state.
            batch: Current training batch.
            context: Pre-update module context.

        Returns:
            Unchanged model state and a batch with doubled increment.
        """
        self._record(f"batch_start:{context.step}")
        return model_state, {"increment": batch["increment"] * 2.0}

    def on_train_batch_end(
        self,
        model_state: Any,
        context: PhiModuleContext,
    ) -> tuple[Any, Mapping[str, Any]]:
        """Replace state and metrics after callback batch-end observation.

        Args:
            model_state: Updated model state.
            context: Post-update module context.

        Returns:
            Incremented model state and a distinguishable total loss.
        """
        self._record(f"batch_end:{context.step}")
        return {"weight": model_state["weight"] + 10.0}, {"train/loss": jnp.asarray(2.0)}

    def on_fit_end(self, model_state: Any, context: PhiModuleContext) -> Any:
        """Increment terminal state after callback fit-end observation.

        Args:
            model_state: Terminal model state.
            context: Terminal module context.

        Returns:
            Model state with its scalar weight incremented by one hundred.
        """
        self._record(f"fit_end:{context.step}")
        return {"weight": model_state["weight"] + 100.0}


class _ContextRecordingCallback(Callback):
    """Capture incoming callback state and metrics around transforming module hooks."""

    def __init__(self) -> None:
        """Initialize empty lifecycle observations."""
        self.fit_start_weight: float | None = None
        self.batch_start_weight: float | None = None
        self.batch_end_weight: float | None = None
        self.batch_end_loss: float | None = None
        self.fit_end_weight: float | None = None
        self.fit_end_loss: float | None = None

    def on_fit_start(self, context: TrainerContext) -> None:
        """Capture state before module fit-start transformation.

        Args:
            context: Incoming fit-start context.
        """
        self.fit_start_weight = float(context.state.model_state["weight"])

    def on_train_batch_start(self, context: TrainerContext) -> None:
        """Capture state before module batch-start transformation.

        Args:
            context: Incoming batch-start context.
        """
        self.batch_start_weight = float(context.state.model_state["weight"])

    def on_train_batch_end(self, context: TrainerContext) -> bool:
        """Capture raw compiled outputs before module batch-end transformation.

        Args:
            context: Incoming batch-end context.

        Returns:
            Always `False`.
        """
        self.batch_end_weight = float(context.state.model_state["weight"])
        self.batch_end_loss = float(context.metrics["train/loss"])
        return False

    def on_fit_end(self, context: TrainerContext) -> None:
        """Capture state and metrics before module fit-end transformation.

        Args:
            context: Incoming fit-end context.
        """
        self.fit_end_weight = float(context.state.model_state["weight"])
        self.fit_end_loss = float(context.metrics["train/loss"])


class _RecordingLogger(ExperimentLogger):
    """Capture trainer logging without external services."""

    def __init__(self) -> None:
        """Initialize empty capture lists."""
        self.parameters: list[Mapping[str, Any]] = []
        self.metrics: list[dict[str, float]] = []
        self.steps: list[int] = []
        self.statuses: list[str] = []

    def log_hyperparameters(self, parameters: Mapping[str, Any]) -> None:
        """Capture resolved parameters.

        Args:
            parameters: Resolved parameters.
        """
        self.parameters.append(parameters)

    def log_metrics(self, metrics: Mapping[str, float], step: int) -> None:
        """Capture the logged optimizer step.

        Args:
            metrics: Scalar metrics.
            step: Optimizer step.
        """
        assert "train/loss" in metrics
        self.metrics.append(dict(metrics))
        self.steps.append(step)

    def log_artifact(self, path: str | Path) -> None:
        """Ignore local artifacts.

        Args:
            path: Artifact path.
        """
        del path

    def finalize(self, status: str) -> None:
        """Capture terminal status.

        Args:
            status: Terminal run status.
        """
        self.statuses.append(status)


class _MemoryCheckpointIO:
    """Provide an in-memory checkpoint backend for trainer routing tests."""

    def __init__(self, restored_state: TrainState, directory: Path | None = None) -> None:
        """Initialize the backend with one restorable state.

        Args:
            restored_state: State returned by full and weights-only restore calls.
            directory: Optional checkpoint-root path exposed to trainer lookup.
        """
        self.restored_state = restored_state
        self.directory = directory
        self.restore_steps: list[int | None] = []
        self.weight_steps: list[int | None] = []
        self.closed = False

    @property
    def latest_step(self) -> int | None:
        """Return the synthetic available checkpoint step.

        Returns:
            Constant checkpoint step `4`.
        """
        return 4

    def save(
        self,
        state: Any,
        step: int,
        metrics: Mapping[str, float] | None = None,
        *,
        force: bool = False,
    ) -> bool:
        """Reject unexpected saves in routing-only tests.

        Args:
            state: Functional state offered for saving.
            step: Checkpoint step.
            metrics: Optional checkpoint metrics.
            force: Whether backend policy bypass was requested.

        Returns:
            Always `False`.
        """
        del state, step, metrics, force
        return False

    def restore(self, target: Any, step: int | None = None) -> TrainState:
        """Record a full restore and return the configured state.

        Args:
            target: Restore structure template.
            step: Requested checkpoint step.

        Returns:
            Configured restored state.
        """
        del target
        self.restore_steps.append(step)
        return self.restored_state

    def restore_weights(self, target: Any, step: int | None = None) -> TrainState:
        """Record a weights-only restore and return the configured state.

        Args:
            target: Fresh target state.
            step: Requested checkpoint step.

        Returns:
            Configured restored state.
        """
        del target
        self.weight_steps.append(step)
        return self.restored_state

    def wait_until_finished(self) -> None:
        """Complete immediately because this backend has no pending writes."""

    def close(self) -> None:
        """Record backend resource closure."""
        self.closed = True


def _state() -> TrainState:
    """Build a minimal valid functional state for trainer tests.

    Returns:
        Training state with one scalar model parameter.
    """
    return TrainState(
        model_state={"weight": jnp.asarray(0.0)},
        optimizer_state=(),
        balancer_state=BalancerState(weights=jnp.ones(1), traces=jnp.zeros(1)),
        rng_key=jax.random.key(1),
        step=jnp.asarray(0, jnp.int32),
        loss_scale=jnp.asarray(1.0, jnp.float32),
        finite_steps=jnp.asarray(0, jnp.int32),
    )


@jax.jit
def _train_step(state: TrainState, batch: dict[str, jax.Array]) -> tuple[TrainState, dict[str, jax.Array]]:
    """Apply one deterministic scalar test update.

    Args:
        state: Current functional training state.
        batch: Mapping containing one scalar increment.

    Returns:
        Updated state and constant loss metric.
    """
    model_state = {"weight": state.model_state["weight"] + batch["increment"]}
    return state.replace(model_state=model_state, step=state.step + 1), {"train/loss": jnp.asarray(1.0)}


def _failing_train_step(state: TrainState, batch: dict[str, jax.Array]) -> tuple[TrainState, dict[str, jax.Array]]:
    """Raise a deterministic training error for exception-order tests.

    Args:
        state: Current functional training state.
        batch: Current training batch.

    Returns:
        No value because this synthetic step always raises.

    Raises:
        RuntimeError: Always.
    """
    del state, batch
    raise RuntimeError("synthetic step failure")


def test_trainer_runs_callbacks_multiple_loggers_and_early_stopping() -> None:
    """Verify the trainer coordinates host services around one reusable compiled step."""
    timeline: list[str] = []
    module = _RecordingModule(timeline)
    callback = _RecordingCallback(timeline)
    early_stopping = EarlyStopping(patience=0)
    first_logger = _RecordingLogger()
    second_logger = _RecordingLogger()
    trainer = Trainer(
        max_steps=5,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(early_stopping, callback),
        loggers=(first_logger, second_logger),
        log_every_n_steps=1,
    )
    result = trainer.fit(
        module,
        _train_step,
        _state(),
        lambda _: {"increment": jnp.asarray(2.0)},
        hyperparameters={"seed": 4},
    )
    assert result.stopped_early is True
    assert result.interrupted is False
    assert result.iterations == 2
    assert int(result.state.step) == 2
    assert float(result.state.model_state["weight"]) == 4.0
    assert callback.events == [
        "setup",
        "fit_start:0",
        "batch_start:0",
        "batch_end:1",
        "batch_start:1",
        "batch_end:2",
        "fit_end:2",
        "teardown",
    ]
    assert module.events == [
        "setup",
        "fit_start:0",
        "batch_start:0",
        "batch_end:1",
        "batch_start:1",
        "batch_end:2",
        "fit_end:2",
        "teardown",
    ]
    assert timeline == [
        "callback:setup",
        "module:setup",
        "callback:fit_start:0",
        "module:fit_start:0",
        "callback:batch_start:0",
        "module:batch_start:0",
        "callback:batch_end:1",
        "module:batch_end:1",
        "callback:batch_start:1",
        "module:batch_start:1",
        "callback:batch_end:2",
        "module:batch_end:2",
        "callback:fit_end:2",
        "module:fit_end:2",
        "callback:teardown",
        "module:teardown",
    ]
    for logger in (first_logger, second_logger):
        assert logger.parameters == [{"seed": 4}]
        assert logger.metrics == [{"train/loss": 1.0}, {"train/loss": 1.0}]
        assert logger.steps == [1, 2]
        assert logger.statuses == ["success"]


def test_learning_rate_monitor_contributes_fit_and_logger_metrics() -> None:
    """Verify schedule values follow completed optimizer steps and enter every standard metric destination."""
    experiment_logger = _RecordingLogger()
    trainer = Trainer(
        max_steps=3,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(LearningRateMonitor(lambda step: 0.1 * 0.5**step, log_key_prefix="train/"),),
        loggers=(experiment_logger,),
        log_every_n_steps=1,
    )

    result = trainer.fit(
        _RecordingModule(),
        _train_step,
        _state(),
        lambda _: {"increment": jnp.asarray(1.0)},
    )

    assert result.metrics["train/lr"] == pytest.approx(0.025)
    assert [metrics["train/lr"] for metrics in experiment_logger.metrics] == pytest.approx([0.1, 0.05, 0.025])


def test_learning_rate_monitor_defaults_to_trainer_logging_cadence() -> None:
    """Verify default LR monitoring evaluates only logged steps and the terminal fit state."""
    evaluated_steps: list[int] = []

    def schedule(step: int) -> float:
        """Record one learning-rate schedule evaluation.

        Args:
            step: Zero-based optimizer schedule step.

        Returns:
            Deterministic learning rate for the requested step.
        """
        evaluated_steps.append(step)
        return 0.1 * 0.5**step

    experiment_logger = _RecordingLogger()
    trainer = Trainer(
        max_steps=4,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(LearningRateMonitor(schedule, log_key_prefix="train/"),),
        loggers=(experiment_logger,),
        log_every_n_steps=3,
    )

    result = trainer.fit(
        _RecordingModule(),
        _train_step,
        _state(),
        lambda _: {"increment": jnp.asarray(1.0)},
    )

    assert evaluated_steps == [0, 2, 3]
    assert [metrics["train/lr"] for metrics in experiment_logger.metrics] == pytest.approx([0.1, 0.025, 0.0125])
    assert result.metrics["train/lr"] == pytest.approx(0.0125)


def test_callbacks_observe_incoming_values_before_module_replacements() -> None:
    """Verify Lightning ordering while preserving explicit module replacements for subsequent lifecycle stages."""
    callback = _ContextRecordingCallback()
    trainer = Trainer(
        max_steps=1,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(callback,),
    )

    result = trainer.fit(
        _TransformingModule(),
        _train_step,
        _state(),
        lambda _: {"increment": jnp.asarray(1.0)},
    )

    assert callback.fit_start_weight == 0.0
    assert callback.batch_start_weight == 1.0
    assert callback.batch_end_weight == 3.0
    assert callback.batch_end_loss == 1.0
    assert callback.fit_end_weight == 13.0
    assert callback.fit_end_loss == 2.0
    assert float(result.state.model_state["weight"]) == 113.0
    assert result.metrics["train/loss"] == 2.0


def test_callback_exception_and_teardown_hooks_run_before_module_hooks() -> None:
    """Verify callback-first Lightning ordering also applies to exceptional cleanup."""
    timeline: list[str] = []
    callback = _RecordingCallback(timeline)
    module = _RecordingModule(timeline)
    trainer = Trainer(
        max_steps=1,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(callback,),
    )
    data_module = MagicMock()

    with pytest.raises(RuntimeError, match="synthetic step failure"):
        trainer.fit(
            module,
            _failing_train_step,
            _state(),
            lambda _: {"increment": jnp.asarray(1.0)},
            datamodule=data_module,
        )

    assert timeline == [
        "callback:setup",
        "module:setup",
        "callback:fit_start:0",
        "module:fit_start:0",
        "callback:batch_start:0",
        "module:batch_start:0",
        "callback:exception:RuntimeError:0",
        "module:exception:RuntimeError:0",
        "callback:teardown",
        "module:teardown",
    ]
    data_module.teardown_stage.assert_called_once_with("fit")


def test_keyboard_interrupt_returns_last_valid_state_and_interrupted_status() -> None:
    """Verify Ctrl+C follows exception hooks, finalizes services, and returns resumable state."""
    timeline: list[str] = []
    callback = _RecordingCallback(timeline)
    module = _RecordingModule(timeline)
    experiment_logger = _RecordingLogger()
    trainer = Trainer(
        max_steps=4,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(callback,),
        loggers=(experiment_logger,),
        log_every_n_steps=1,
    )

    def batches(iteration: int) -> dict[str, jax.Array]:
        """Return two batches before simulating a user interruption.

        Args:
            iteration: Zero-based requested batch index.

        Returns:
            One scalar update batch.

        Raises:
            KeyboardInterrupt: When the third batch is requested.
        """
        if iteration == 2:
            raise KeyboardInterrupt
        return {"increment": jnp.asarray(1.0)}

    result = trainer.fit(module, _train_step, _state(), batches)

    assert result.interrupted is True
    assert result.stopped_early is False
    assert result.iterations == 2
    assert int(result.state.step) == 2
    assert result.metrics == {"train/loss": 1.0}
    assert trainer.interrupted is True
    assert trainer.received_sigterm is False
    assert experiment_logger.statuses == ["interrupted"]
    assert timeline[-4:] == [
        "callback:exception:KeyboardInterrupt:2",
        "module:exception:KeyboardInterrupt:2",
        "callback:teardown",
        "module:teardown",
    ]


def test_sigterm_terminates_fit_after_interrupted_lifecycle_cleanup() -> None:
    """Verify `SIGTERM` checkpoints through exception hooks and remains terminal to caller orchestration."""
    timeline: list[str] = []
    callback = _RecordingCallback(timeline)
    module = _RecordingModule(timeline)
    experiment_logger = _RecordingLogger()
    trainer = Trainer(
        max_steps=1,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(callback,),
        loggers=(experiment_logger,),
    )

    def terminated_batch_source(iteration: int) -> dict[str, jax.Array]:
        """Simulate the trainer signal handler terminating before a batch is returned.

        Args:
            iteration: Requested zero-based iteration, which must be zero.

        Raises:
            SystemExit: Always, using the standard `SIGTERM` exit status.
        """
        assert iteration == 0
        trainer.received_sigterm = True
        raise SystemExit(128 + signal.SIGTERM)

    with pytest.raises(SystemExit) as exception:
        trainer.fit(module, _train_step, _state(), terminated_batch_source)

    assert exception.value.code == 128 + signal.SIGTERM
    assert trainer.interrupted is True
    assert experiment_logger.statuses == ["interrupted"]
    assert timeline[-4:] == [
        "callback:exception:SystemExit:0",
        "module:exception:SystemExit:0",
        "callback:teardown",
        "module:teardown",
    ]


def test_fit_signal_handler_converts_sigterm_and_restores_previous_handlers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify `SIGTERM` becomes a graceful interruption and signal state is process-local.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    trainer = Trainer(max_steps=1, strategy=SingleDeviceStrategy(jax.devices("cpu")[0]))
    installed: dict[int, Any] = {}
    previous = {signal.SIGINT: object(), signal.SIGTERM: object()}
    monkeypatch.setattr(signal, "getsignal", lambda signum: previous[signum])
    monkeypatch.setattr(signal, "signal", lambda signum, handler: installed.__setitem__(signum, handler))
    handler = _FitSignalHandler(trainer)

    handler.install()
    with pytest.raises(SystemExit) as exception:
        installed[signal.SIGTERM](signal.SIGTERM, None)
    handler.restore()

    assert exception.value.code == 128 + signal.SIGTERM
    assert trainer.received_sigterm is True
    assert installed == previous


def test_trainer_prints_bfloat16_amp_and_selected_cuda_environment(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify trainer diagnostics distinguish mixed precision, GPU selection, and TPU availability.

    Args:
        capsys: Pytest fixture capturing runtime diagnostics.
        monkeypatch: Pytest attribute patch helper.
    """
    gpu = SimpleNamespace(
        platform="gpu",
        process_index=0,
        id=0,
        platform_version="CUDA 13.0",
        device_kind="NVIDIA RTX",
    )
    strategy = cast(Strategy, SimpleNamespace(is_global_zero=True, devices=(gpu,)))
    trainer = Trainer(max_steps=1, precision="bf16-mixed", matmul_precision="default", strategy=strategy)
    monkeypatch.setattr(jax, "devices", lambda: (gpu,))

    trainer.print_environment_info()

    assert capsys.readouterr().out.splitlines() == [
        "",
        "Using bfloat16 Automatic Mixed Precision (AMP)",
        "Matmul precision: default",
        "GPU available: True (cuda), used: True",
        "TPU available: False, using: 0 TPU cores",
        "",
    ]


def test_trainer_environment_supports_devices_without_optional_backend_metadata(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify environment reporting tolerates the minimal metadata exposed by some JAX GPU plugins.

    Args:
        capsys: Pytest fixture capturing runtime diagnostics.
        monkeypatch: Pytest attribute patch helper.
    """
    gpu = SimpleNamespace(platform="gpu", process_index=0, id=0)
    strategy = cast(Strategy, SimpleNamespace(is_global_zero=True, devices=(gpu,)))
    trainer = Trainer(max_steps=1, matmul_precision="default", strategy=strategy)
    monkeypatch.setattr(jax, "devices", lambda: (gpu,))

    trainer.print_environment_info()

    assert capsys.readouterr().out.splitlines() == [
        "",
        "Using 32-bit true precision",
        "Matmul precision: default",
        "GPU available: True (gpu), used: True",
        "TPU available: False, using: 0 TPU cores",
        "",
    ]


def test_trainer_applies_and_restores_matmul_precision_for_fit_and_prediction() -> None:
    """Verify explicit matmul precision covers lazy execution without leaking process-global state."""
    observed: list[tuple[str, Any]] = []

    def recording_step(state: TrainState, batch: dict[str, jax.Array]) -> tuple[TrainState, dict[str, jax.Array]]:
        """Record fit-time JAX precision before applying one synthetic update.

        Args:
            state: Current training state.
            batch: Synthetic update batch.

        Returns:
            Updated state and scalar metrics.
        """
        observed.append(("fit", jax.config.jax_default_matmul_precision))
        return _train_step(state, batch)

    class RecordingPredictionModule(_RecordingModule):
        """Record JAX precision from prediction computation."""

        def predict_step(self, model_state: Any, batch: dict[str, jax.Array]) -> jax.Array:
            """Record and evaluate one prediction batch.

            Args:
                model_state: Explicit scalar model state.
                batch: Prediction batch containing model inputs.

            Returns:
                Batched scalar model outputs.
            """
            observed.append(("predict", jax.config.jax_default_matmul_precision))
            return super().predict_step(model_state, batch)

    previous_precision = jax.config.jax_default_matmul_precision
    trainer = Trainer(
        max_steps=1,
        matmul_precision="highest",
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
    )
    trainer.fit(_RecordingModule(), recording_step, _state(), ({"increment": jnp.asarray(1.0)},))
    trainer.predict(
        RecordingPredictionModule(),
        _state().replace(model_state={"weight": jnp.asarray(2.0)}),
        ({"inputs": jnp.asarray([[1.0]])},),
    )

    assert observed == [("fit", "highest"), ("predict", "highest")]
    assert jax.config.jax_default_matmul_precision == previous_precision


def test_trainer_null_matmul_precision_preserves_external_policy() -> None:
    """Verify a null trainer option does not override an externally configured JAX policy."""
    observed: list[Any] = []

    def recording_step(state: TrainState, batch: dict[str, jax.Array]) -> tuple[TrainState, dict[str, jax.Array]]:
        """Record the ambient JAX policy before one synthetic update.

        Args:
            state: Current training state.
            batch: Synthetic update batch.

        Returns:
            Updated state and scalar metrics.
        """
        observed.append(jax.config.jax_default_matmul_precision)
        return _train_step(state, batch)

    trainer = Trainer(max_steps=1, strategy=SingleDeviceStrategy(jax.devices("cpu")[0]))
    with jax.default_matmul_precision("high"):
        trainer.fit(_RecordingModule(), recording_step, _state(), ({"increment": jnp.asarray(1.0)},))

    assert observed == ["high"]


def test_trainer_rejects_unknown_matmul_precision() -> None:
    """Verify the trainer rejects unsupported matrix-multiplication precision names."""
    with pytest.raises(ValueError, match="`matmul_precision`"):
        Trainer(max_steps=1, matmul_precision="float32")


class _PreparableSource:
    """Record the device selected by Trainer-owned source preparation."""

    def __init__(self) -> None:
        """Initialize without a selected device."""
        self.device: Any | None = None
        self.steps: list[int] = []

    def prepare(self, device: Any) -> "_PreparableSource":
        """Record and preserve the Trainer-selected device.

        Args:
            device: Strategy root device.

        Returns:
            This recording source.
        """
        self.device = device
        return self

    def __call__(self, step: int) -> dict[str, np.ndarray]:
        """Return one host increment after Trainer-owned preparation.

        Args:
            step: Requested global optimizer step.

        Returns:
            Mapping containing one floating scalar increment.

        Raises:
            RuntimeError: If the Trainer did not prepare this source before sampling.
        """
        if self.device is None:
            raise RuntimeError("Source was sampled before Trainer preparation.")
        self.steps.append(step)
        return {"increment": np.asarray(1.0, dtype=np.float64)}


def test_trainer_prepares_sources_and_batches_for_its_strategy() -> None:
    """Verify source preparation, precision casting, placement, and non-floating preservation are centralized."""
    device = jax.devices("cpu")[0]
    trainer = Trainer(
        max_steps=1,
        precision="32-true",
        strategy=SingleDeviceStrategy(device),
    )
    source = _PreparableSource()

    assert trainer.prepare_batch_source(source) is source
    assert source.device == device

    def plain_source(step: int) -> dict[str, int]:
        """Return one unprepared scalar mapping.

        Args:
            step: Requested synthetic step.

        Returns:
            Mapping containing `step`.
        """
        return {"step": step}

    assert trainer.prepare_batch_source(plain_source) is plain_source

    batch = trainer.prepare_batch(
        {
            "inputs": np.ones((2, 1), dtype=np.float64),
            "mask": np.asarray([True, False]),
        }
    )
    assert batch["inputs"].dtype == jnp.float32
    assert batch["mask"].dtype == jnp.bool_
    assert batch["inputs"].devices() == {device}
    assert batch["mask"].devices() == {device}

    fit_source = _PreparableSource()
    result = trainer.fit(_RecordingModule(), _train_step, _state(), fit_source)
    assert fit_source.device == device
    assert fit_source.steps == [0]
    assert int(result.state.step) == 1


def test_trainer_builds_training_source_from_data_module() -> None:
    """Verify a training plan lets the Trainer own DataModule source construction and placement."""
    device = jax.devices("cpu")[0]
    trainer = Trainer(max_steps=1, strategy=SingleDeviceStrategy(device))
    source = _PreparableSource()
    data_module = MagicMock(spec=PhiDataModule)
    data_module.train_batch_source.return_value = source
    sampling_key = jax.random.key(17)
    plan = TrainingPlan(_train_step, ("train",))

    result = trainer.fit(
        _RecordingModule(),
        plan,
        _state(),
        datamodule=data_module,
        sampling_key=sampling_key,
    )

    data_module.prepare_stage.assert_called_once_with("fit")
    requested_keys, requested_sampling_key = data_module.train_batch_source.call_args.args
    assert requested_keys == ("train",)
    np.testing.assert_array_equal(jax.random.key_data(requested_sampling_key), jax.random.key_data(sampling_key))
    data_module.teardown_stage.assert_called_once_with("fit")
    assert source.device == device
    assert source.steps == [0]
    assert int(result.state.step) == 1


class _ScalarObjective:
    """Provide one differentiable scalar loss for precision-aware trainer assembly tests."""

    @property
    def loss_names(self) -> tuple[str, ...]:
        """Return the scalar loss name.

        Returns:
            One stable loss name.
        """
        return ("loss",)

    def losses(
        self,
        model_apply: Any,
        model_state: Any,
        batches: dict[str, dict[str, jax.Array]],
    ) -> dict[str, jax.Array]:
        """Evaluate a squared scalar prediction loss.

        Args:
            model_apply: Pure scalar model application callable.
            model_state: Explicit scalar parameter mapping.
            batches: Nested input batch mapping.

        Returns:
            Mapping containing one scalar loss.
        """
        prediction = model_apply(model_state, batches["data"]["inputs"])
        return {"loss": jnp.mean(prediction**2)}


def test_trainer_builds_consistent_mixed_precision_state_and_step() -> None:
    """Verify trainer helper methods reuse one FP16 mixed-precision policy."""

    def model_apply(model_state: dict[str, jax.Array], inputs: jax.Array) -> jax.Array:
        """Apply one scalar weight.

        Args:
            model_state: Scalar weight mapping.
            inputs: Input vector.

        Returns:
            Weighted inputs.
        """
        return model_state["weight"] * inputs

    trainer = Trainer(
        max_steps=1,
        precision="16-mixed",
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
    )
    balancer = StaticLossBalancer(("loss",))
    optimizer = optax.sgd(1.0e-2)
    state = trainer.initialize_state(
        {"weight": jnp.asarray(1.0)},
        optimizer,
        balancer.initialize(),
        jax.random.key(4),
    )
    module = PhiModule(model_apply, _ScalarObjective())
    train_step = trainer.compile_train_step(module, balancer, optimizer)
    state, metrics = train_step(state, {"data": {"inputs": jnp.ones((2,), jnp.float32)}})
    assert int(state.step) == 1
    assert float(state.loss_scale) == 32768.0
    assert float(metrics["train/precision/gradients_finite"]) == 1.0


def test_trainer_routes_restore_and_close_through_model_checkpoint() -> None:
    """Verify trainer restore helpers use the configured checkpoint callback backend."""
    restored_state = _state().replace(step=jnp.asarray(4, jnp.int32))
    requested_batch_steps: list[int] = []
    checkpoint_io = _MemoryCheckpointIO(restored_state)
    checkpoint = ModelCheckpoint(checkpoint_io, save_last=False)
    trainer = Trainer(
        max_steps=1,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(checkpoint,),
    )

    def batch_source(step: int) -> dict[str, jax.Array]:
        """Record the global step requested after checkpoint restoration.

        Args:
            step: Restored global optimizer step.

        Returns:
            Synthetic scalar update batch.
        """
        requested_batch_steps.append(step)
        return {"increment": jnp.asarray(1.0)}

    result = trainer.resume_latest(
        _RecordingModule(),
        _train_step,
        _state(),
        batch_source,
    )
    loaded = trainer.load_weights(_state(), step=3)
    trainer.close()

    assert checkpoint_io.restore_steps == [4]
    assert requested_batch_steps == [4]
    assert checkpoint_io.weight_steps == [3]
    assert int(result.state.step) == 5
    assert loaded is restored_state
    assert checkpoint_io.closed is True


def test_trainer_context_manager_closes_resources_idempotently() -> None:
    """Verify a trainer context releases checkpoint resources once and tolerates repeated cleanup."""
    checkpoint_io = _MemoryCheckpointIO(_state())
    checkpoint = ModelCheckpoint(checkpoint_io, save_last=False)

    with Trainer(
        max_steps=1,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(checkpoint,),
    ) as trainer:
        assert trainer._closed is False

    trainer.close()

    assert trainer._closed is True
    assert checkpoint_io.closed is True


def test_trainer_resolves_the_latest_checkpoint_root(tmp_path: Path) -> None:
    """Verify post-fit orchestration can retrieve the latest committed checkpoint.

    Args:
        tmp_path: Temporary checkpoint-root path.
    """
    checkpoint_io = _MemoryCheckpointIO(_state(), tmp_path / "checkpoints")
    trainer = Trainer(
        max_steps=1,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(ModelCheckpoint(checkpoint_io),),
    )

    checkpoint_path, checkpoint_step = trainer.latest_checkpoint()

    assert checkpoint_path == (tmp_path / "checkpoints").resolve()
    assert checkpoint_step == 4


def test_trainer_fit_and_predict_restore_checkpoint_state_internally(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify Lightning-style checkpoint arguments are resolved inside both trainer execution APIs.

    Args:
        monkeypatch: Pytest attribute patch helper.
        tmp_path: Temporary checkpoint-root placeholder.
    """
    calls: list[tuple[Path, bool, int | None]] = []

    def restore(
        target: TrainState,
        ckpt_path: str | Path | None,
        *,
        weights_only: bool,
        step: int | None,
    ) -> TrainState:
        """Record restoration policy and return distinguishable state.

        Args:
            target: Fresh functional restore template.
            ckpt_path: Configured checkpoint root.
            weights_only: Whether only model weights are requested.
            step: Optional exact checkpoint step.

        Returns:
            State with a restored step or model weight.
        """
        assert ckpt_path is not None
        calls.append((Path(ckpt_path), weights_only, step))
        if weights_only:
            return target.replace(model_state={"weight": jnp.asarray(3.0)})
        return target.replace(step=jnp.asarray(4, jnp.int32))

    monkeypatch.setattr(trainer_module, "restore_checkpoint", restore)
    trainer = Trainer(max_steps=1, strategy=SingleDeviceStrategy(jax.devices("cpu")[0]))
    fit_data_module = MagicMock()
    prediction_data_module = MagicMock()
    requested_steps: list[int] = []

    def batches(step: int) -> dict[str, jax.Array]:
        """Record the restored optimizer step used for sampling.

        Args:
            step: Global step requested by :meth:`Trainer.fit`.

        Returns:
            One scalar training update.
        """
        requested_steps.append(step)
        return {"increment": jnp.asarray(1.0)}

    fit_result = trainer.fit(
        _RecordingModule(),
        _train_step,
        _state(),
        batches,
        ckpt_path=tmp_path,
        ckpt_step=4,
        datamodule=fit_data_module,
    )
    predictions = trainer.predict(
        _RecordingModule(),
        _state(),
        ({"inputs": jnp.asarray([[2.0]])},),
        ckpt_path=tmp_path,
        ckpt_step=5,
        datamodule=prediction_data_module,
    )

    assert calls == [(tmp_path, False, 4), (tmp_path, True, 5)]
    assert requested_steps == [4]
    assert int(fit_result.state.step) == 5
    np.testing.assert_array_equal(predictions, [[6.0]])
    fit_data_module.teardown_stage.assert_called_once_with("fit")
    prediction_data_module.teardown_stage.assert_called_once_with("predict")


def test_trainer_rejects_multiple_model_checkpoint_callbacks() -> None:
    """Verify checkpoint restore ownership remains unambiguous."""
    checkpoint_io = _MemoryCheckpointIO(_state())
    callback = ModelCheckpoint(checkpoint_io)
    with pytest.raises(ValueError, match="Only one"):
        Trainer(max_steps=1, callbacks=(callback, callback))


def test_trainer_rejects_multiple_prediction_writer_callbacks(tmp_path: Path) -> None:
    """Verify prediction artifact ownership remains unambiguous.

    Args:
        tmp_path: Temporary writer destination directory.
    """
    callback = PredictionWriter(tmp_path, save_file_name="prediction")
    with pytest.raises(ValueError, match="Only one `PredictionWriter`"):
        Trainer(max_steps=1, callbacks=(callback, callback))


def test_trainer_predict_concatenates_only_valid_padded_rows() -> None:
    """Verify prediction uses the module API and removes padded rows before concatenation."""
    timeline: list[str] = []
    callback = _RecordingCallback(timeline)
    trainer = Trainer(
        max_steps=1,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(callback,),
    )
    module = _RecordingModule(timeline)
    batches = (
        {"inputs": jnp.asarray([[1.0], [2.0]]), "mask": jnp.asarray([True, True])},
        {"inputs": jnp.asarray([[3.0], [0.0]]), "mask": jnp.asarray([True, False])},
    )

    state = _state().replace(model_state={"weight": jnp.asarray(2.0)})
    predictions = trainer.predict(module, state, batches)

    np.testing.assert_array_equal(predictions, [[2.0], [4.0], [6.0]])
    assert timeline == [
        "callback:setup",
        "module:setup",
        "callback:predict_start",
        "module:predict_start",
        "callback:predict_epoch_start",
        "module:predict_epoch_start",
        "callback:predict_batch_start:0",
        "module:predict_batch_start:0",
        "callback:predict_batch_end:0",
        "module:predict_batch_end:0",
        "callback:predict_batch_start:1",
        "module:predict_batch_start:1",
        "callback:predict_batch_end:1",
        "module:predict_batch_end:1",
        "callback:predict_epoch_end",
        "module:predict_epoch_end",
        "callback:predict_end",
        "module:predict_end",
        "callback:teardown",
        "module:teardown",
    ]


def test_trainer_predict_sets_up_and_uses_data_module_source() -> None:
    """Verify omitted batches are obtained through the trainer-owned prediction DataModule lifecycle."""
    pool = HostPool(
        inputs=np.asarray([[1.0], [2.0]], dtype=np.float32),
        targets=np.empty((2, 0), dtype=np.float32),
        aux={},
        metadata={},
        reference_shape=(2,),
        flat_index=np.arange(2),
    )
    data_module = MagicMock(spec=PhiDataModule)
    data_module.predict_batch_source.return_value = ChunkedPredictionSource(pool, 2)
    trainer = Trainer(max_steps=1, strategy=SingleDeviceStrategy(jax.devices("cpu")[0]))
    state = _state().replace(model_state={"weight": jnp.asarray(2.0)})

    predictions = trainer.predict(_RecordingModule(), state, datamodule=data_module)

    np.testing.assert_array_equal(predictions, [[2.0], [4.0]])
    data_module.prepare_stage.assert_called_once_with("predict")
    data_module.predict_batch_source.assert_called_once_with()
    data_module.teardown_stage.assert_called_once_with("predict")


def test_trainer_predict_skips_missing_data_module_source() -> None:
    """Verify a DataModule without prediction data produces a clean no-op lifecycle."""
    callback = _RecordingCallback()
    data_module = MagicMock(spec=PhiDataModule)
    data_module.predict_batch_source.return_value = None
    trainer = Trainer(
        max_steps=1,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(callback,),
    )

    predictions = trainer.predict(_RecordingModule(), _state(), datamodule=data_module)

    assert predictions is None
    assert callback.events == []
    data_module.prepare_stage.assert_called_once_with("predict")
    data_module.teardown_stage.assert_called_once_with("predict")


def test_trainer_prediction_context_exposes_source_pool_and_rank() -> None:
    """Verify prediction callbacks receive reconstruction data and distributed ownership."""

    def model_apply(model_state: dict[str, jax.Array], inputs: jax.Array) -> jax.Array:
        """Apply one scalar prediction weight.

        Args:
            model_state: Scalar weight mapping.
            inputs: One input batch.

        Returns:
            Weighted input batch.
        """
        return model_state["weight"] * inputs

    pool = HostPool(
        inputs=np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32),
        targets=np.empty((3, 0), dtype=np.float32),
        aux={},
        metadata={},
        reference_shape=(3,),
        flat_index=np.arange(3),
    )
    callback = _PredictionContextCallback()
    trainer = Trainer(
        max_steps=1,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(callback,),
    )
    module = PhiModule(model_apply, _ScalarObjective())

    state = _state().replace(model_state={"weight": jnp.asarray(2.0)})
    trainer.predict(module, state, ChunkedPredictionSource(pool, 2))

    assert callback.context is not None
    assert callback.context.pool is pool
    assert callback.context.is_global_zero is True
    assert callback.context.total_batches == 2
    assert callback.context.batch is None


def test_trainer_predict_can_stream_without_retaining_predictions() -> None:
    """Verify callbacks receive batch outputs while host prediction collection is disabled."""

    def model_apply(model_state: dict[str, jax.Array], inputs: jax.Array) -> jax.Array:
        """Apply one scalar prediction weight.

        Args:
            model_state: Scalar weight mapping.
            inputs: One input batch.

        Returns:
            Weighted input batch.
        """
        return model_state["weight"] * inputs

    timeline: list[str] = []
    callback = _RecordingCallback(timeline)
    trainer = Trainer(
        max_steps=1,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(callback,),
    )
    module = PhiModule(model_apply, _ScalarObjective())

    result = trainer.predict(
        module,
        _state().replace(model_state={"weight": jnp.asarray(2.0)}),
        ({"inputs": jnp.asarray([[1.0], [2.0]])},),
        return_predictions=False,
    )

    assert result is None
    assert "callback:predict_batch_end:0" in timeline
    assert timeline[-2:] == ["callback:predict_end", "callback:teardown"]
