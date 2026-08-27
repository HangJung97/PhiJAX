import logging
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import pytest

from phijax.callbacks import ModelCheckpoint, TrainerContext


class _RecordingCheckpointIO:
    """Record checkpoint backend operations without filesystem access."""

    def __init__(self) -> None:
        """Initialize empty save records and resource-state flags."""
        self.saved: list[tuple[Any, int, dict[str, float], bool]] = []
        self.waited = False
        self.closed = False

    @property
    def latest_step(self) -> int | None:
        """Return the last recorded step.

        Returns:
            Latest saved step, or `None` before the first save.
        """
        return self.saved[-1][1] if self.saved else None

    def save(
        self,
        state: Any,
        step: int,
        metrics: Mapping[str, float] | None = None,
        *,
        force: bool = False,
    ) -> bool:
        """Record one checkpoint request.

        Args:
            state: Functional state to record.
            step: Checkpoint step.
            metrics: Host scalar metrics.
            force: Whether policy bypass was requested.

        Returns:
            Always `True`.
        """
        self.saved.append((state, step, dict(metrics or {}), force))
        return True

    def restore(self, target: Any, step: int | None = None) -> Any:
        """Return the supplied restore target.

        Args:
            target: Restore target.
            step: Optional restore step.

        Returns:
            Unchanged `target`.
        """
        del step
        return target

    def restore_weights(self, target: Any, step: int | None = None) -> Any:
        """Return the supplied weight-loading target.

        Args:
            target: Weight-loading target.
            step: Optional restore step.

        Returns:
            Unchanged `target`.
        """
        del step
        return target

    def wait_until_finished(self) -> None:
        """Record that pending writes were awaited."""
        self.waited = True

    def close(self) -> None:
        """Record backend closure."""
        self.closed = True


def test_model_checkpoint_saves_periodic_and_terminal_states(caplog: pytest.LogCaptureFixture) -> None:
    """Verify checkpoint scheduling and scalar metric transfer are callback-owned."""
    checkpoint_io = _RecordingCheckpointIO()
    callback = ModelCheckpoint(checkpoint_io, every_n_steps=2, save_last=True)
    with caplog.at_level(logging.INFO, logger="phijax.callbacks.model_checkpoint"):
        callback.setup()
        assert callback.on_train_batch_end(TrainerContext("one", 1, {"loss": jnp.asarray(3.0)})) is False
        assert callback.on_train_batch_end(TrainerContext("two", 2, {"loss": jnp.asarray(2.0)})) is False
        callback.on_fit_end(TrainerContext("three", 3, {"loss": jnp.asarray(1.0)}))
        callback.teardown()
        callback.close()

    assert checkpoint_io.saved == [
        ("two", 2, {"loss": 2.0}, True),
        ("three", 3, {"loss": 1.0}, True),
    ]
    assert [record.message for record in caplog.records] == [
        "Checkpoint submitted at step 2.",
        "Checkpoint submitted at step 3.",
    ]
    assert checkpoint_io.waited is True
    assert checkpoint_io.closed is True


def test_model_checkpoint_validates_interval_and_scalar_metrics() -> None:
    """Verify invalid scheduling and non-scalar checkpoint metadata fail clearly."""
    checkpoint_io = _RecordingCheckpointIO()
    with pytest.raises(ValueError, match="every_n_steps"):
        ModelCheckpoint(checkpoint_io, every_n_steps=0)
    callback = ModelCheckpoint(checkpoint_io, every_n_steps=1)
    with pytest.raises(ValueError, match="scalar"):
        callback.on_train_batch_end(TrainerContext(None, 1, {"loss": jnp.ones(2)}))


def test_model_checkpoint_optionally_saves_last_valid_state_on_exception() -> None:
    """Verify exceptional shutdown checkpointing is explicit and deduplicates the current step."""
    checkpoint_io = _RecordingCheckpointIO()
    callback = ModelCheckpoint(checkpoint_io, save_last=False, save_on_exception=True)
    context = TrainerContext("valid", 7, {"loss": jnp.asarray(0.5)})

    callback.setup()
    callback.on_exception(KeyboardInterrupt(), context)
    callback.on_exception(RuntimeError("duplicate"), context)

    assert checkpoint_io.saved == [("valid", 7, {"loss": 0.5}, True)]
