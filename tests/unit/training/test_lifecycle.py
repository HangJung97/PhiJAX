import logging
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import pytest

from phijax.callbacks import Callback, CallbackContext, PredictionContext
from phijax.core import BasePhiModule, PhiModuleContext
from phijax.training.lifecycle import TaskLifecycle
from phijax.training.loggers import ExperimentLogger
from phijax.types import NamedBatches


class _LifecycleCallback(Callback):
    """Record callback resource lifecycle events."""

    def __init__(self, timeline: list[str]) -> None:
        """Store a shared event timeline.

        Args:
            timeline: Mutable event list used by assertions.
        """
        self.timeline = timeline

    def setup(self) -> None:
        """Record callback setup."""
        self.timeline.append("callback:setup")

    def on_exception(self, exception: BaseException, context: CallbackContext) -> None:
        """Record callback exception dispatch.

        Args:
            exception: Simulated prediction failure.
            context: Most recent prediction context.
        """
        del exception, context
        self.timeline.append("callback:exception")

    def teardown(self) -> None:
        """Record callback teardown."""
        self.timeline.append("callback:teardown")


class _LifecycleModule(BasePhiModule):
    """Record module resource lifecycle events."""

    def __init__(self, timeline: list[str]) -> None:
        """Initialize the module and shared timeline.

        Args:
            timeline: Mutable event list used by assertions.
        """
        super().__init__()
        self.timeline = timeline

    @property
    def loss_names(self) -> Sequence[str]:
        """Return one synthetic loss name.

        Returns:
            Single loss-name tuple.
        """
        return ("loss",)

    def forward(self, model_state: Any, inputs: jax.Array) -> jax.Array:
        """Return inputs unchanged.

        Args:
            model_state: Unused model state.
            inputs: Synthetic input batch.

        Returns:
            Unchanged inputs.
        """
        del model_state
        return inputs

    def training_step(self, model_state: Any, batches: NamedBatches) -> Mapping[str, jax.Array]:
        """Return one unused synthetic loss.

        Args:
            model_state: Unused model state.
            batches: Unused named batches.

        Returns:
            Scalar loss mapping.
        """
        del model_state, batches
        return {"loss": jax.numpy.asarray(0.0)}

    def setup(self) -> None:
        """Record module setup."""
        self.timeline.append("module:setup")

    def on_exception(self, exception: BaseException, context: PhiModuleContext) -> None:
        """Record module exception dispatch.

        Args:
            exception: Simulated prediction failure.
            context: Most recent module context.
        """
        del exception, context
        self.timeline.append("module:exception")

    def teardown(self) -> None:
        """Record module teardown."""
        self.timeline.append("module:teardown")


class _LifecycleLogger(ExperimentLogger):
    """Record terminal logger statuses."""

    def __init__(self) -> None:
        """Initialize an empty status list."""
        self.setup_calls = 0
        self.statuses: list[str] = []

    def setup(self) -> None:
        """Record one resource setup call."""
        self.setup_calls += 1

    def finalize(self, status: str) -> None:
        """Record one terminal status.

        Args:
            status: Terminal task status.
        """
        self.statuses.append(status)


def test_task_lifecycle_preserves_lightning_order_and_idempotent_finalization() -> None:
    """Verify shared setup, exception, logger, and teardown semantics."""
    timeline: list[str] = []
    callback = _LifecycleCallback(timeline)
    module = _LifecycleModule(timeline)
    logger = _LifecycleLogger()
    lifecycle = TaskLifecycle((callback,), module, logger, is_global_zero=True)
    prediction_context = PredictionContext(outputs=None, batch_index=None, metadata={})
    module_context = PhiModuleContext(step=0, metrics={})
    error = RuntimeError("failure")

    lifecycle.setup()
    lifecycle.handle_exception(error, prediction_context, module_context)
    lifecycle.finalize("failed")
    lifecycle.finalize("success")
    lifecycle.teardown()

    assert lifecycle.callbacks == (callback,)
    assert timeline == [
        "callback:setup",
        "module:setup",
        "callback:exception",
        "module:exception",
        "callback:teardown",
        "module:teardown",
    ]
    assert logger.statuses == ["failed"]
    assert logger.setup_calls == 1


def test_task_lifecycle_does_not_touch_logger_resources_outside_global_rank() -> None:
    """Verify nonzero ranks retain logger presence without owning its resources."""
    timeline: list[str] = []
    logger = _LifecycleLogger()
    lifecycle = TaskLifecycle((), _LifecycleModule(timeline), logger, is_global_zero=False)

    lifecycle.setup()
    lifecycle.finalize("success")
    lifecycle.teardown()

    assert logger.setup_calls == 0
    assert logger.statuses == []


class _FailingCallback(_LifecycleCallback):
    """Fail in selected cleanup hooks after recording them."""

    def __init__(self, timeline: list[str], *, fail_on: frozenset[str]) -> None:
        """Store the timeline and the hooks that raise.

        Args:
            timeline: Mutable event list used by assertions.
            fail_on: Hook names, `exception` or `teardown`, that raise after recording.
        """
        super().__init__(timeline)
        self.fail_on = fail_on

    def on_exception(self, exception: BaseException, context: CallbackContext) -> None:
        """Record exception dispatch, then optionally fail.

        Args:
            exception: Task failure.
            context: Most recent context.

        Raises:
            ValueError: If `exception` is in `fail_on`.
        """
        super().on_exception(exception, context)
        if "exception" in self.fail_on:
            raise ValueError("on_exception failed")

    def teardown(self) -> None:
        """Record teardown, then optionally fail.

        Raises:
            OSError: If `teardown` is in `fail_on`.
        """
        super().teardown()
        if "teardown" in self.fail_on:
            raise OSError("teardown failed")


class _FailingLogger(_LifecycleLogger):
    """Record terminal statuses, then fail."""

    def finalize(self, status: str) -> None:
        """Record the status, then fail.

        Args:
            status: Terminal task status.

        Raises:
            ConnectionError: Always.
        """
        super().finalize(status)
        raise ConnectionError("finalize failed")


def test_task_lifecycle_runs_all_cleanup_after_a_task_error_and_keeps_that_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verify failing cleanup hooks are logged and noted on the task error instead of replacing it.

    Args:
        caplog: Pytest log capture fixture.
    """
    timeline: list[str] = []
    failing = _FailingCallback(timeline, fail_on=frozenset({"exception", "teardown"}))
    logger = _FailingLogger()
    lifecycle = TaskLifecycle(
        (failing, _LifecycleCallback(timeline)), _LifecycleModule(timeline), logger, is_global_zero=True
    )
    error = RuntimeError("task failed")

    lifecycle.setup()
    with caplog.at_level(logging.ERROR, logger="phijax.training.lifecycle"):
        lifecycle.handle_exception(error, PredictionContext(None, None, {}), PhiModuleContext(step=0, metrics={}))
        lifecycle.finalize("failed")
        lifecycle.teardown()

    assert timeline == [
        "callback:setup",
        "callback:setup",
        "module:setup",
        "callback:exception",
        "callback:exception",
        "module:exception",
        "callback:teardown",
        "callback:teardown",
        "module:teardown",
    ]
    assert logger.statuses == ["failed"]
    notes = error.__notes__
    assert len(notes) == 3
    assert "_FailingCallback.on_exception also failed" in notes[0]
    assert "_FailingLogger.finalize also failed" in notes[1]
    assert "_FailingCallback.teardown also failed" in notes[2]
    assert caplog.text.count("failed during task cleanup") == 3


def test_task_lifecycle_tears_down_everything_before_raising_after_success() -> None:
    """Verify a successful task still tears down every resource, then raises the first teardown failure."""
    timeline: list[str] = []
    first = _FailingCallback(timeline, fail_on=frozenset({"teardown"}))
    second = _FailingCallback(timeline, fail_on=frozenset({"teardown"}))
    lifecycle = TaskLifecycle((first, second), _LifecycleModule(timeline), _LifecycleLogger(), is_global_zero=True)
    lifecycle.setup()
    lifecycle.finalize("success")

    with pytest.raises(OSError, match="teardown failed") as raised:
        lifecycle.teardown()

    assert timeline[-3:] == ["callback:teardown", "callback:teardown", "module:teardown"]
    assert raised.value.__notes__ == ["Another teardown hook also failed: OSError('teardown failed')"]
