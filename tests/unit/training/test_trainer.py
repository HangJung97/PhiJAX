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

import phijax.training.loops.fit_loop as fit_loop_module
import phijax.training.trainer as trainer_module
from phijax.balancers import BalancerState, BalancerUpdatePlan, StaticLossBalancer
from phijax.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    PredictionContext,
    PredictionWriter,
    TrainerContext,
)
from phijax.core import BasePhiModule, PhiModule, PhiModuleContext
from phijax.data import ChunkedPredictionSource, DataStage, HostPool, NamedBatchSource, PhiDataModule
from phijax.models import InitializedModel
from phijax.training import (
    ExperimentLogger,
    SingleDeviceStrategy,
    Strategy,
    Trainer,
    TrainingPlan,
    TrainState,
    build_training_plan,
)
from phijax.training.connectors.signal_connector import _SignalConnector


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


class _LoggingModule(_RecordingModule):
    """Customize host-side destinations for compiled causal-style diagnostics."""

    def on_train_batch_end(
        self,
        model_state: Any,
        context: PhiModuleContext,
    ) -> tuple[Any, Mapping[str, Any]]:
        """Expose one diagnostic in progress and suppress another from loggers.

        Args:
            model_state: Updated model state.
            context: Post-update module context containing compiled diagnostics.

        Returns:
            Unchanged model state and metrics.
        """
        self.log(
            "train/causal/active_window",
            context.metrics["train/causal/active_window"],
            prog_bar=True,
        )
        self.log(
            "train/precision/loss_scale",
            context.metrics["train/precision/loss_scale"],
            logger=False,
        )
        self.log(
            "train/causal/window_weights",
            context.metrics["train/causal/window_weights"],
            logger=False,
            prog_bar=False,
        )
        self.log("train/causal/host_metric", jnp.asarray(3.0))
        return model_state, context.metrics


class _InvalidLoggingModule(_RecordingModule):
    """Exercise invalid host-side module logging declarations."""

    def __init__(self, case: str) -> None:
        """Initialize one invalid logging case.

        Args:
            case: Validation case selected by the test.
        """
        super().__init__()
        self.case = case

    def on_train_batch_end(
        self,
        model_state: Any,
        context: PhiModuleContext,
    ) -> tuple[Any, Mapping[str, Any]]:
        """Submit the configured invalid metric declaration.

        Args:
            model_state: Updated model state.
            context: Post-update module context.

        Returns:
            Unchanged model state and metrics when validation unexpectedly succeeds.
        """
        if self.case == "name_type":
            self.log(1, jnp.asarray(1.0))  # type: ignore[arg-type]
        elif self.case == "empty_name":
            self.log(" ", jnp.asarray(1.0))
        elif self.case == "logger_type":
            self.log("invalid", jnp.asarray(1.0), logger=1)  # type: ignore[arg-type]
        elif self.case == "prog_bar_type":
            self.log("invalid", jnp.asarray(1.0), prog_bar=1)  # type: ignore[arg-type]
        else:
            self.log("invalid", jnp.ones(2))
        return model_state, context.metrics


class _ContextRecordingCallback(Callback):
    """Capture incoming callback state and metrics around transforming module hooks."""

    def __init__(self) -> None:
        """Initialize empty lifecycle observations."""
        self.fit_start_weight: float | None = None
        self.batch_start_weight: float | None = None
        self.batch_start_increment: float | None = None
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
        assert context.batch is not None
        self.batch_start_increment = float(context.batch["increment"])

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

    def log_hyperparams(self, parameters: Mapping[str, Any]) -> None:
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
        self.open_calls = 0
        self.close_calls = 0
        self.opened = False
        self.closed = False
        self.callback_states: dict[str, Mapping[str, Any]] = {}

    def open(self) -> None:
        """Record backend activation while allowing repeated Trainer stages."""
        if self.opened:
            return
        self.open_calls += 1
        self.opened = True
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
        callback_states: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> bool:
        """Reject unexpected saves in routing-only tests.

        Args:
            state: Functional state offered for saving.
            step: Checkpoint step.
            metrics: Optional checkpoint metrics.
            force: Whether backend policy bypass was requested.
            callback_states: Optional persistent callback state.

        Returns:
            Always `False`.
        """
        del state, step, metrics, force, callback_states
        return False

    @property
    def steps(self) -> tuple[int, ...]:
        """Return the synthetic available checkpoint steps.

        Returns:
            One constant checkpoint step.
        """
        return (4,)

    def checkpoint_path(self, step: int) -> Path | None:
        """Return a synthetic checkpoint path when a directory is configured.

        Args:
            step: Requested checkpoint step.

        Returns:
            Step path or `None` for memory-only storage.
        """
        return None if self.directory is None else self.directory / str(step)

    def delete(self, step: int) -> None:
        """Reject unexpected deletion in routing-only tests.

        Args:
            step: Checkpoint step offered for deletion.

        Raises:
            AssertionError: Always, because these tests do not exercise retention.
        """
        raise AssertionError(f"Unexpected deletion of checkpoint {step}.")

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

    def restore_callback_states(self, step: int | None = None) -> dict[str, Mapping[str, Any]]:
        """Return configured persistent callback state.

        Args:
            step: Requested checkpoint step.

        Returns:
            Copy of configured callback state.
        """
        del step
        return dict(self.callback_states)

    def wait_until_finished(self) -> None:
        """Complete immediately because this backend has no pending writes."""

    def close(self) -> None:
        """Record backend resource closure."""
        if not self.opened:
            return
        self.close_calls += 1
        self.opened = False
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
        sampling_key=jax.random.key(2),
        balancer_key=jax.random.key(3),
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


def _balancer_train_step(state: TrainState, batch: dict[str, jax.Array]) -> tuple[TrainState, dict[str, jax.Array]]:
    """Apply one synthetic update and expose the active balancer weight.

    Args:
        state: Current functional training state.
        batch: Mapping containing one scalar model increment.

    Returns:
        Updated state and the balancer weight used by this step.
    """
    model_state = {"weight": state.model_state["weight"] + batch["increment"]}
    return state.replace(model_state=model_state, step=state.step + 1), {"train/loss": state.balancer_state.weights[0]}


@pytest.mark.parametrize(
    ("initial_step", "update_start_step", "expected_updates"),
    [(0, 0, [0, 2]), (0, 2, [2]), (0, 1, [1]), (5, 0, [6])],
)
def test_trainer_schedules_adaptive_balancer_updates_from_host_steps(
    initial_step: int,
    update_start_step: int,
    expected_updates: list[int],
) -> None:
    """Verify current-batch updates use absolute restored steps without reading device state each iteration.

    Args:
        initial_step: Restored optimizer step before fitting.
        update_start_step: Absolute step anchoring the update cadence.
        expected_updates: Absolute steps at which the update must run.
    """
    update_steps: list[int] = []

    def update(
        model_state: dict[str, jax.Array],
        batches: dict[str, jax.Array],
        balancer_state: BalancerState,
    ) -> BalancerState:
        """Record the selected batch and increment the synthetic weight.

        Args:
            model_state: Unused explicit model state.
            batches: Current training batch containing its absolute step.
            balancer_state: Current synthetic balancer state.

        Returns:
            Balancer state with its weight incremented once.
        """
        del model_state
        update_steps.append(int(batches["source_step"]))
        return BalancerState(weights=balancer_state.weights + 1.0, traces=balancer_state.traces)

    update_plan = BalancerUpdatePlan(
        update,
        every_n_steps=2,
        update_start_step=update_start_step,
    )
    state = _state().replace(step=jnp.asarray(initial_step, jnp.int32))
    trainer = Trainer(
        max_steps=3,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
    )

    result = trainer.fit_state(
        _RecordingModule(),
        TrainingPlan(_balancer_train_step, balancer_update=update_plan),
        state,
        lambda step: {"increment": jnp.asarray(1.0), "source_step": jnp.asarray(step)},
    )

    assert update_steps == expected_updates
    assert int(result.state.step) == initial_step + 3
    np.testing.assert_array_equal(result.state.balancer_state.weights, [1.0 + len(expected_updates)])


def test_trainer_samples_fixed_balancer_diagnostics_once_from_restored_state() -> None:
    """Verify fixed adaptive diagnostics use the restored key and remain separate from current batches."""

    class SampleableSource:
        """Provide current training batches and record one fixed diagnostic sample."""

        def __init__(self) -> None:
            """Initialize empty diagnostic sampling records."""
            self.sample_calls: list[tuple[jax.Array, Mapping[str, int]]] = []

        def __call__(self, step: int) -> dict[str, jax.Array]:
            """Return one current training batch.

            Args:
                step: Absolute host optimizer step.

            Returns:
                Scalar training batch derived from `step`.
            """
            return {"increment": jnp.asarray(float(step + 1))}

        def sample(self, key: jax.Array, batch_sizes: Mapping[str, int]) -> dict[str, jax.Array]:
            """Record and return a fixed diagnostic batch.

            Args:
                key: Diagnostic sampling key folded with the restored step.
                batch_sizes: Requested objective batch sizes.

            Returns:
                Fixed scalar diagnostic batch.
            """
            self.sample_calls.append((key, batch_sizes))
            return {"increment": jnp.asarray(9.0)}

    diagnostic_values: list[float] = []

    def update(
        model_state: dict[str, jax.Array],
        batches: dict[str, jax.Array],
        balancer_state: BalancerState,
    ) -> BalancerState:
        """Record the fixed diagnostic value and update the synthetic weight.

        Args:
            model_state: Unused explicit model state.
            batches: Fixed diagnostic batch.
            balancer_state: Current synthetic balancer state.

        Returns:
            Balancer state whose weight accumulates the diagnostic value.
        """
        del model_state
        diagnostic_values.append(float(batches["increment"]))
        return BalancerState(
            weights=balancer_state.weights + batches["increment"],
            traces=balancer_state.traces,
        )

    source = SampleableSource()
    state = _state().replace(step=jnp.asarray(3, jnp.int32))
    update_plan = BalancerUpdatePlan(
        update,
        every_n_steps=1,
        update_start_step=0,
        batch_sizes={"train": 2},
    )
    trainer = Trainer(
        max_steps=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
    )

    result = trainer.fit_state(
        _RecordingModule(),
        TrainingPlan(_balancer_train_step, balancer_update=update_plan),
        state,
        source,
    )

    assert diagnostic_values == [9.0, 9.0]
    assert len(source.sample_calls) == 1
    sampled_key, batch_sizes = source.sample_calls[0]
    expected_key = jax.random.fold_in(state.balancer_key, 3)
    np.testing.assert_array_equal(jax.random.key_data(sampled_key), jax.random.key_data(expected_key))
    assert batch_sizes == {"train": 2}
    np.testing.assert_array_equal(result.state.balancer_state.weights, [19.0])


def test_trainer_reads_device_step_only_at_fit_boundaries(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify ordinary iterations use the host counter instead of transferring `TrainState.step`.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    state_reads: list[int] = []
    read_state_step = fit_loop_module._state_step

    def record_state_step(state: TrainState) -> int:
        """Record one explicit device-to-host step transfer.

        Args:
            state: Functional training state at a fit boundary.

        Returns:
            Host integer returned by the production helper.
        """
        step = read_state_step(state)
        state_reads.append(step)
        return step

    monkeypatch.setattr(trainer_module, "_state_step", record_state_step)
    monkeypatch.setattr(fit_loop_module, "_state_step", record_state_step)
    trainer = Trainer(
        max_steps=4,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
    )

    result = trainer.fit_state(
        _RecordingModule(),
        _train_step,
        _state(),
        lambda _: {"increment": jnp.asarray(1.0)},
    )

    assert state_reads == [0, 4]
    assert result.iterations == 4


def test_trainer_does_not_synchronize_unused_metrics_at_logging_cadence(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify disabled host consumers leave intermediate device metrics asynchronous.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    synchronization_calls: list[Any] = []
    block_until_ready = fit_loop_module.jax.block_until_ready

    def record_synchronization(value: Any) -> Any:
        """Record and perform one explicit Trainer synchronization.

        Args:
            value: Metric PyTree being synchronized.

        Returns:
            Synchronized input value.
        """
        synchronization_calls.append(value)
        return block_until_ready(value)

    monkeypatch.setattr(fit_loop_module.jax, "block_until_ready", record_synchronization)
    trainer = Trainer(
        max_steps=4,
        log_every_n_steps=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
    )

    trainer.fit_state(
        _RecordingModule(),
        _train_step,
        _state(),
        lambda _: {"increment": jnp.asarray(1.0)},
    )

    assert len(synchronization_calls) == 1


def test_trainer_rejects_custom_step_without_one_step_increment() -> None:
    """Verify host and device progress divergence fails before successful fit finalization."""

    def invalid_step(state: TrainState, batch: dict[str, jax.Array]) -> tuple[TrainState, dict[str, jax.Array]]:
        """Return metrics without advancing the explicit optimizer step.

        Args:
            state: Current functional training state.
            batch: Unused synthetic batch.

        Returns:
            Unchanged state and one scalar metric.
        """
        del batch
        return state, {"train/loss": jnp.asarray(1.0)}

    trainer = Trainer(
        max_steps=2,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
    )

    with pytest.raises(RuntimeError, match=r"must increment `TrainState\.step` exactly once"):
        trainer.fit_state(
            _RecordingModule(),
            invalid_step,
            _state(),
            lambda _: {"increment": jnp.asarray(1.0)},
        )


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
    result = trainer.fit_state(
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
        callbacks=(LearningRateMonitor(lambda step: 0.1 * 0.5**step, optimizer_name="Adam"),),
        loggers=(experiment_logger,),
        log_every_n_steps=1,
    )

    result = trainer.fit_state(
        _RecordingModule(),
        _train_step,
        _state(),
        lambda _: {"increment": jnp.asarray(1.0)},
    )

    assert result.metrics["optimizer/lr-Adam"] == pytest.approx(0.025)
    assert [metrics["optimizer/lr-Adam"] for metrics in experiment_logger.metrics] == pytest.approx([0.1, 0.05, 0.025])


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
        callbacks=(LearningRateMonitor(schedule, optimizer_name="Adam"),),
        loggers=(experiment_logger,),
        log_every_n_steps=3,
    )

    result = trainer.fit_state(
        _RecordingModule(),
        _train_step,
        _state(),
        lambda _: {"increment": jnp.asarray(1.0)},
    )

    assert evaluated_steps == [0, 2, 3]
    assert [metrics["optimizer/lr-Adam"] for metrics in experiment_logger.metrics] == pytest.approx(
        [0.1, 0.025, 0.0125]
    )
    assert result.metrics["optimizer/lr-Adam"] == pytest.approx(0.0125)


def test_learning_rate_monitor_stops_fit_before_first_update_without_logger() -> None:
    """Verify logger validation runs before the first training batch is requested."""
    requested_steps: list[int] = []

    def batches(step: int) -> dict[str, jax.Array]:
        """Record an unexpected batch request.

        Args:
            step: Requested global optimizer step.

        Returns:
            Synthetic scalar batch.
        """
        requested_steps.append(step)
        return {"increment": jnp.asarray(1.0)}

    trainer = Trainer(
        max_steps=1,
        callbacks=(LearningRateMonitor(lambda step: step, optimizer_name="Adam"),),
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
    )
    with pytest.raises(RuntimeError, match="Cannot use `LearningRateMonitor` with a Trainer that has no logger"):
        trainer.fit_state(_RecordingModule(), _train_step, _state(), batches)

    assert requested_steps == []


def test_trainer_routes_causal_diagnostics_to_logger_not_default_progress() -> None:
    """Verify unmatched diagnostics stay logger-only while total loss remains visible."""
    experiment_logger = _RecordingLogger()

    def diagnostic_step(state: TrainState, batch: dict[str, jax.Array]) -> tuple[TrainState, dict[str, jax.Array]]:
        """Return total loss and one causal-style scalar diagnostic.

        Args:
            state: Current functional state.
            batch: Synthetic scalar update batch.

        Returns:
            Incremented state and stable scalar metrics.
        """
        next_state, metrics = _train_step(state, batch)
        return next_state, {
            **metrics,
            "train/loss/pde/heat": jnp.asarray(0.5),
            "train/weight/pde/heat": jnp.asarray(1.0),
            "train/diagnostic/causal/mean_weight": jnp.asarray(0.75),
            "train/diagnostic/causal/window_weights": jnp.asarray([0.5, 1.0]),
        }

    trainer = Trainer(
        max_steps=1,
        logger=experiment_logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
    )
    trainer.fit_state(
        _RecordingModule(),
        diagnostic_step,
        _state(),
        ({"increment": jnp.asarray(1.0)},),
    )

    assert set(trainer.progress_bar_metrics) == {
        "train/loss",
        "train/loss/pde/heat",
        "train/weight/pde/heat",
    }
    assert set(trainer.logged_metrics) == {
        "train/loss",
        "train/loss/pde/heat",
        "train/weight/pde/heat",
        "train/diagnostic/causal/mean_weight",
    }
    assert "train/diagnostic/causal/window_weights" in trainer.callback_metrics
    assert experiment_logger.metrics[-1]["train/diagnostic/causal/mean_weight"] == pytest.approx(0.75)


def test_module_log_overrides_destinations_and_adds_host_metrics() -> None:
    """Verify host module declarations override defaults without hiding callback diagnostics."""
    experiment_logger = _RecordingLogger()

    def diagnostic_step(state: TrainState, batch: dict[str, jax.Array]) -> tuple[TrainState, dict[str, jax.Array]]:
        """Return scalar and array diagnostics for host-side routing.

        Args:
            state: Current functional state.
            batch: Synthetic scalar update batch.

        Returns:
            Incremented state and stable diagnostic mapping.
        """
        next_state, metrics = _train_step(state, batch)
        return next_state, {
            **metrics,
            "train/causal/active_window": jnp.asarray(2.0),
            "train/causal/window_weights": jnp.asarray([0.5, 1.0]),
            "train/precision/loss_scale": jnp.asarray(1.0),
        }

    trainer = Trainer(
        max_steps=1,
        logger=experiment_logger,
        enable_progress_bar=False,
        enable_model_summary=False,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
    )
    trainer.fit_state(
        _LoggingModule(),
        diagnostic_step,
        _state(),
        ({"increment": jnp.asarray(1.0)},),
    )

    assert set(trainer.progress_bar_metrics) == {"train/loss", "train/causal/active_window"}
    assert set(trainer.logged_metrics) == {
        "train/loss",
        "train/causal/active_window",
        "train/causal/host_metric",
    }
    assert set(trainer.callback_metrics) == {
        "train/loss",
        "train/causal/active_window",
        "train/causal/window_weights",
        "train/precision/loss_scale",
        "train/causal/host_metric",
    }
    assert "train/precision/loss_scale" not in experiment_logger.metrics[-1]
    assert experiment_logger.metrics[-1]["train/causal/host_metric"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("case", "error", "message"),
    [
        ("name_type", TypeError, "name.*string"),
        ("empty_name", ValueError, "name.*non-empty"),
        ("logger_type", TypeError, "logger.*boolean"),
        ("prog_bar_type", TypeError, "prog_bar.*boolean"),
        ("array", ValueError, "must be scalar"),
    ],
)
def test_module_log_validates_host_declarations(
    case: str,
    error: type[Exception],
    message: str,
) -> None:
    """Verify invalid declarations fail during the module hook and release the collector.

    Args:
        case: Invalid declaration selected by the parameterization.
        error: Expected exception type.
        message: Expected exception message pattern.
    """
    module = _InvalidLoggingModule(case)
    trainer = Trainer(
        max_steps=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
    )
    with pytest.raises(error, match=message):
        trainer.fit_state(module, _train_step, _state(), ({"increment": jnp.asarray(1.0)},))
    with pytest.raises(RuntimeError, match="only during `on_train_batch_end`"):
        module.log("outside", jnp.asarray(1.0))


def test_callbacks_observe_incoming_values_before_module_replacements() -> None:
    """Verify Lightning ordering while preserving explicit module replacements for subsequent lifecycle stages."""
    callback = _ContextRecordingCallback()
    trainer = Trainer(
        max_steps=1,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(callback,),
    )

    result = trainer.fit_state(
        _TransformingModule(),
        _train_step,
        _state(),
        lambda _: {"increment": jnp.asarray(1.0)},
    )

    assert callback.fit_start_weight == 0.0
    assert callback.batch_start_weight == 1.0
    assert callback.batch_start_increment == 1.0
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
        trainer.fit_state(
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

    result = trainer.fit_state(module, _train_step, _state(), batches)

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
        trainer.fit_state(module, _train_step, _state(), terminated_batch_source)

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
    installed: dict[int, Any] = {}
    previous = {signal.SIGINT: object(), signal.SIGTERM: object()}
    monkeypatch.setattr(signal, "getsignal", lambda signum: previous[signum])
    monkeypatch.setattr(signal, "signal", lambda signum, handler: installed.__setitem__(signum, handler))
    handler = _SignalConnector()

    handler.install()
    with pytest.raises(SystemExit) as exception:
        installed[signal.SIGTERM](signal.SIGTERM, None)
    handler.restore()

    assert exception.value.code == 128 + signal.SIGTERM
    assert handler.received_sigterm is True
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
    monkeypatch.setattr(jax, "devices", lambda: (gpu,))
    Trainer(max_steps=1, precision="bf16-mixed", matmul_precision="default", strategy=strategy)

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
    monkeypatch.setattr(jax, "devices", lambda: (gpu,))
    Trainer(max_steps=1, matmul_precision="default", strategy=strategy)

    assert capsys.readouterr().out.splitlines() == [
        "",
        "Using 32-bit true precision",
        "Matmul precision: default",
        "GPU available: True (gpu), used: True",
        "TPU available: False, using: 0 TPU cores",
        "",
    ]


def test_trainer_does_not_print_environment_on_nonzero_rank(capsys: pytest.CaptureFixture[str]) -> None:
    """Verify automatic environment reporting is restricted to global rank zero.

    Args:
        capsys: Pytest fixture capturing runtime diagnostics.
    """
    strategy = cast(Strategy, SimpleNamespace(is_global_zero=False, devices=()))

    Trainer(max_steps=1, strategy=strategy)

    assert capsys.readouterr().out == ""


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
    trainer.fit_state(_RecordingModule(), recording_step, _state(), ({"increment": jnp.asarray(1.0)},))
    trainer.predict_state(
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
        trainer.fit_state(_RecordingModule(), recording_step, _state(), ({"increment": jnp.asarray(1.0)},))

    assert observed == ["high"]


def test_trainer_rejects_unknown_matmul_precision() -> None:
    """Verify the trainer rejects unsupported matrix-multiplication precision names."""
    with pytest.raises(ValueError, match="`matmul_precision`"):
        Trainer(max_steps=1, matmul_precision="float32")


def test_trainer_rejects_removed_deterministic_argument() -> None:
    """Verify deterministic execution is not represented by an inert Trainer argument."""
    with pytest.raises(TypeError, match="unexpected keyword argument 'deterministic'"):
        cast(Any, Trainer)(max_steps=1, deterministic=True)


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
    result = trainer.fit_state(_RecordingModule(), _train_step, _state(), fit_source)
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
    state = _state()
    plan = TrainingPlan(_train_step, ("train",))

    result = trainer.fit_state(
        _RecordingModule(),
        plan,
        state,
        datamodule=data_module,
    )

    data_module.prepare_stage.assert_called_once_with("fit")
    requested_keys, requested_sampling_key = data_module.train_batch_source.call_args.args
    assert requested_keys == ("train",)
    np.testing.assert_array_equal(jax.random.key_data(requested_sampling_key), jax.random.key_data(state.sampling_key))
    data_module.teardown_stage.assert_called_once_with("fit")
    assert source.device == device
    assert source.steps == [0]
    assert int(result.state.step) == 1


def test_trainer_restores_sampling_state_before_building_data_module_source(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify resumed DataModule sampling uses the checkpoint key and global-step offset.

    Args:
        monkeypatch: Pytest attribute patch helper.
        tmp_path: Synthetic checkpoint-root path.
    """
    restored_key = jax.random.key(19)
    restored_state = _state().replace(step=jnp.asarray(5, jnp.int32), sampling_key=restored_key)
    source = _PreparableSource()
    data_module = MagicMock(spec=PhiDataModule)
    data_module.train_batch_source.return_value = source
    trainer = Trainer(max_steps=1, strategy=SingleDeviceStrategy(jax.devices("cpu")[0]))
    monkeypatch.setattr(trainer, "_restore_state", lambda *args, **kwargs: restored_state)

    result = trainer.fit_state(
        _RecordingModule(),
        TrainingPlan(_train_step, ("train",)),
        _state(),
        ckpt_path=tmp_path,
        datamodule=data_module,
    )

    _, sampling_key = data_module.train_batch_source.call_args.args
    np.testing.assert_array_equal(jax.random.key_data(sampling_key), jax.random.key_data(restored_key))
    assert source.steps == [5]
    assert int(result.state.step) == 6


class _ScalarObjective:
    """Provide one differentiable scalar loss for precision-aware trainer assembly tests."""

    @property
    def loss_names(self) -> tuple[str, ...]:
        """Return the scalar loss name.

        Returns:
            One stable loss name.
        """
        return ("loss",)

    @property
    def batch_keys(self) -> tuple[str, ...]:
        """Return the synthetic input batch name.

        Returns:
            One stable batch key.
        """
        return ("data",)

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


class _ConciseDataModule(PhiDataModule):
    """Provide deterministic finite batches for concise Trainer API tests."""

    def setup(self, stage: DataStage) -> None:
        """Construct one scalar input pool for the requested stage.

        Args:
            stage: Requested `fit` or `predict` stage.
        """
        inputs = np.asarray([[1.0], [2.0], [3.0]], dtype=np.float32)
        self.pools = {"data": HostPool(inputs)} if stage == "fit" else {"predict": HostPool(inputs)}

    def train_batch_source(self, batch_keys: tuple[str, ...], key: jax.Array) -> NamedBatchSource:
        """Build one deterministic all-row source.

        Args:
            batch_keys: Objective batch names requested by the Trainer.
            key: Explicit persistent sampling key.

        Returns:
            Finite-row source aligned with the objective.
        """
        return NamedBatchSource.from_pools(self._require_setup("fit"), {"data": "all"}, key, names=batch_keys)

    def predict_batch_source(self) -> ChunkedPredictionSource:
        """Build one ordered prediction source.

        Returns:
            Prediction source covering the complete scalar pool.
        """
        pool = self.prediction_pool()
        if pool is None:
            raise RuntimeError("Prediction data is unavailable.")
        return ChunkedPredictionSource(pool, 2)

    def prediction_pool(self) -> HostPool | None:
        """Return the prepared prediction pool.

        Returns:
            Prepared prediction pool, or `None` during fitting.
        """
        return self.pools.get("predict")


def _scalar_model_factory(
    *,
    key: jax.Array,
    input_mean: jax.typing.ArrayLike | None,
    input_std: jax.typing.ArrayLike | None,
    precision: Any,
) -> InitializedModel:
    """Initialize one scalar model through the public factory contract.

    Args:
        key: Model-parameter initialization key.
        input_mean: Optional input mean, expected to be absent in this fixture.
        input_std: Optional input standard deviation, expected to be absent in this fixture.
        precision: Trainer precision policy.

    Returns:
        Pure scalar model with one randomly initialized weight.
    """
    del precision
    assert input_mean is None
    assert input_std is None

    def apply(state: dict[str, jax.Array], inputs: jax.Array) -> jax.Array:
        """Apply the scalar model weight.

        Args:
            state: Mapping containing one scalar weight.
            inputs: Scalar model inputs.

        Returns:
            Weighted inputs.
        """
        return state["weight"] * inputs

    return InitializedModel(apply, {"weight": jax.random.normal(key, ())})


def _concise_module() -> PhiModule:
    """Build one unbound concise-path module blueprint.

    Returns:
        PhiModule containing the scalar factory and objective.
    """
    return PhiModule(_scalar_model_factory, _ScalarObjective())


def test_concise_fit_is_deterministic_for_integer_and_typed_key_seeds() -> None:
    """Verify equivalent root seeds produce identical split state, batches, metrics, and parameters."""
    first_blueprint = _concise_module()
    second_blueprint = _concise_module()
    trainer = Trainer(max_steps=2, strategy=SingleDeviceStrategy(jax.devices("cpu")[0]))

    first = trainer.fit(
        first_blueprint,
        datamodule=_ConciseDataModule(),
        optimizer=optax.sgd(1.0e-2),
        seed=7,
    )
    second = trainer.fit(
        second_blueprint,
        datamodule=_ConciseDataModule(),
        optimizer=optax.sgd(1.0e-2),
        seed=jax.random.key(7),
    )
    model_key, runtime_key, sampling_key, balancer_key = jax.random.split(jax.random.key(7), 4)
    explicit_blueprint = _concise_module()
    explicit_module, model_state = explicit_blueprint.prepare_model(
        key=model_key,
        input_mean=None,
        input_std=None,
        precision=trainer.precision,
    )
    explicit_balancer = StaticLossBalancer(explicit_module.loss_names)
    explicit_optimizer = optax.sgd(1.0e-2)
    explicit_plan = build_training_plan(
        trainer,
        explicit_module,
        explicit_balancer,
        explicit_optimizer,
    )
    explicit_state = trainer.initialize_state(
        model_state,
        explicit_optimizer,
        explicit_balancer.initialize(),
        runtime_key,
        sampling_key=sampling_key,
        balancer_key=balancer_key,
    )
    explicit = trainer.fit_state(
        explicit_module,
        explicit_plan,
        explicit_state,
        datamodule=_ConciseDataModule(),
    )

    for comparison in (second, explicit):
        for first_leaf, comparison_leaf in zip(
            jax.tree.leaves(first.state),
            jax.tree.leaves(comparison.state),
            strict=True,
        ):
            if jax.dtypes.issubdtype(first_leaf.dtype, jax.dtypes.prng_key):
                np.testing.assert_array_equal(jax.random.key_data(first_leaf), jax.random.key_data(comparison_leaf))
            else:
                np.testing.assert_array_equal(first_leaf, comparison_leaf)
        assert first.metrics == comparison.metrics
    assert first.iterations == second.iterations == explicit.iterations == 2
    assert first.module is not first_blueprint
    with pytest.raises(RuntimeError, match="uninitialized"):
        first_blueprint.forward(first.state.model_state, jnp.ones((1, 1)))


def test_concise_fit_defaults_to_equal_static_weights_and_predicts_from_result() -> None:
    """Verify the common path assembles static balancing and binds post-fit prediction state."""
    trainer = Trainer(max_steps=1, strategy=SingleDeviceStrategy(jax.devices("cpu")[0]))
    data_module = _ConciseDataModule()

    result = trainer.fit(
        _concise_module(),
        datamodule=data_module,
        optimizer=optax.sgd(1.0e-2),
        seed=3,
    )
    predictions = trainer.predict(result, datamodule=data_module)

    assert isinstance(result.module, PhiModule)
    np.testing.assert_array_equal(result.state.balancer_state.weights, np.ones(1, dtype=np.float32))
    assert predictions is not None
    assert predictions.shape == (3, 1)
    np.testing.assert_allclose(
        predictions,
        np.asarray([[1.0], [2.0], [3.0]]) * np.asarray(result.state.model_state["weight"]),
    )


def test_concise_fit_tears_down_data_after_model_initialization_failure() -> None:
    """Verify Trainer releases a prepared DataModule when a lazy model factory fails."""

    def failing_factory(**runtime: Any) -> InitializedModel:
        """Fail after receiving Trainer-owned model initialization values.

        Args:
            **runtime: Initialization key, input statistics, and precision.

        Returns:
            No initialized model because construction always fails.

        Raises:
            RuntimeError: Always.
        """
        del runtime
        raise RuntimeError("model initialization failed")

    data_module = _ConciseDataModule()
    trainer = Trainer(max_steps=1, strategy=SingleDeviceStrategy(jax.devices("cpu")[0]))

    with pytest.raises(RuntimeError, match="model initialization failed"):
        trainer.fit(
            PhiModule(failing_factory, _ScalarObjective()),
            datamodule=data_module,
            optimizer=optax.sgd(1.0e-2),
            seed=0,
        )

    assert data_module.prepared_stage is None


def test_concise_fit_validates_seed_and_balancer_contracts() -> None:
    """Verify invalid root keys and balancer loss-name ordering fail before updates."""
    trainer = Trainer(max_steps=1, strategy=SingleDeviceStrategy(jax.devices("cpu")[0]))
    module = _concise_module()
    optimizer = optax.sgd(1.0e-2)

    with pytest.raises(TypeError, match="integer or an unbatched"):
        trainer.fit(module, datamodule=_ConciseDataModule(), optimizer=optimizer, seed=True)
    with pytest.raises(TypeError, match="integer or an unbatched"):
        trainer.fit(module, datamodule=_ConciseDataModule(), optimizer=optimizer, seed=jnp.ones(2))
    with pytest.raises(ValueError, match="exactly match"):
        trainer.fit(
            module,
            datamodule=_ConciseDataModule(),
            optimizer=optimizer,
            seed=0,
            balancer=StaticLossBalancer(("other",)),
        )


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
    blueprint = PhiModule(InitializedModel(model_apply, {}), _ScalarObjective())
    module, _ = blueprint.prepare_model(
        key=jax.random.key(0),
        input_mean=jnp.zeros(1),
        input_std=jnp.ones(1),
        precision=trainer.precision,
    )
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
    checkpoint_io.callback_states = trainer.callback_state_dict()

    def batch_source(step: int) -> dict[str, jax.Array]:
        """Record the global step requested after checkpoint restoration.

        Args:
            step: Restored global optimizer step.

        Returns:
            Synthetic scalar update batch.
        """
        requested_batch_steps.append(step)
        return {"increment": jnp.asarray(1.0)}

    result = trainer.fit_state(
        _RecordingModule(),
        _train_step,
        _state(),
        batch_source,
        ckpt_path="last",
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
        checkpoint_io.open()

    trainer.close()

    assert checkpoint_io.close_calls == 1


def test_trainer_automatically_closes_and_reopens_checkpoint_resources() -> None:
    """Verify repeated fit stages require no explicit Trainer cleanup."""
    checkpoint_io = _MemoryCheckpointIO(_state())
    trainer = Trainer(
        max_steps=1,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(ModelCheckpoint(checkpoint_io, save_last=False),),
    )

    first = trainer.fit_state(_RecordingModule(), _train_step, _state(), ({"increment": jnp.asarray(1.0)},))
    second = trainer.fit_state(_RecordingModule(), _train_step, first.state, ({"increment": jnp.asarray(1.0)},))

    assert int(second.state.step) == 2
    assert checkpoint_io.open_calls == 2
    assert checkpoint_io.close_calls == 2
    assert checkpoint_io.closed is True


def test_trainer_closes_checkpoint_resources_after_fit_failure() -> None:
    """Verify exceptional fit termination closes an opened checkpoint backend."""
    checkpoint_io = _MemoryCheckpointIO(_state())
    trainer = Trainer(
        max_steps=2,
        strategy=SingleDeviceStrategy(jax.devices("cpu")[0]),
        callbacks=(ModelCheckpoint(checkpoint_io, save_last=False),),
    )

    def failing_source(step: int) -> dict[str, jax.Array]:
        """Return one batch before simulating a source failure.

        Args:
            step: Requested optimizer step.

        Returns:
            Synthetic first-step batch.

        Raises:
            RuntimeError: When the Trainer requests the second batch.
        """
        if step > 0:
            raise RuntimeError("batch failure")
        return {"increment": jnp.asarray(1.0)}

    with pytest.raises(RuntimeError, match="batch failure"):
        trainer.fit_state(_RecordingModule(), _train_step, _state(), failing_source)

    assert checkpoint_io.open_calls == 1
    assert checkpoint_io.close_calls == 1
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

    def restore_state(
        target: TrainState,
        ckpt_path: str | Path | None,
        ckpt_step: int | None,
        *,
        weights_only: bool,
        restore_callbacks: bool = False,
    ) -> TrainState:
        """Exercise the patched state loader without filesystem callback metadata.

        Args:
            target: Fresh functional restore template.
            ckpt_path: Configured checkpoint root.
            ckpt_step: Optional exact checkpoint step.
            weights_only: Whether only model weights are requested.
            restore_callbacks: Whether callback state would be restored in production.

        Returns:
            State returned by the patched checkpoint helper.
        """
        del restore_callbacks
        return restore(target, ckpt_path, weights_only=weights_only, step=ckpt_step)

    monkeypatch.setattr(trainer, "_restore_state", restore_state)
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

    fit_result = trainer.fit_state(
        _RecordingModule(),
        _train_step,
        _state(),
        batches,
        ckpt_path=tmp_path,
        ckpt_step=4,
        datamodule=fit_data_module,
    )
    predictions = trainer.predict_state(
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
    predictions = trainer.predict_state(module, state, batches)

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

    predictions = trainer.predict_state(_RecordingModule(), state, datamodule=data_module)

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

    predictions = trainer.predict_state(_RecordingModule(), _state(), datamodule=data_module)

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
    blueprint = PhiModule(InitializedModel(model_apply, {}), _ScalarObjective())
    module, _ = blueprint.prepare_model(
        key=jax.random.key(0),
        input_mean=jnp.zeros(1),
        input_std=jnp.ones(1),
        precision=trainer.precision,
    )

    state = _state().replace(model_state={"weight": jnp.asarray(2.0)})
    trainer.predict_state(module, state, ChunkedPredictionSource(pool, 2))

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
    blueprint = PhiModule(InitializedModel(model_apply, {}), _ScalarObjective())
    module, _ = blueprint.prepare_model(
        key=jax.random.key(0),
        input_mean=jnp.zeros(1),
        input_std=jnp.ones(1),
        precision=trainer.precision,
    )

    result = trainer.predict_state(
        module,
        _state().replace(model_state={"weight": jnp.asarray(2.0)}),
        ({"inputs": jnp.asarray([[1.0], [2.0]])},),
        return_predictions=False,
    )

    assert result is None
    assert "callback:predict_batch_end:0" in timeline
    assert timeline[-2:] == ["callback:predict_end", "callback:teardown"]
