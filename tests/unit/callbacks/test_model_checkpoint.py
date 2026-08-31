import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from phijax.callbacks import ModelCheckpoint, TrainerContext


class _RecordingCheckpointIO:
    """Record checkpoint backend operations without filesystem access."""

    def __init__(self) -> None:
        """Initialize empty save records and resource-state flags."""
        self.saved: list[tuple[Any, int, dict[str, float], bool, dict[str, Mapping[str, Any]]]] = []
        self.deleted: list[int] = []
        self.waited = False
        self.closed = False
        self.opened = False

    def open(self) -> None:
        """Record backend activation for a fit stage."""
        self.opened = True
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
        callback_states: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> bool:
        """Record one checkpoint request.

        Args:
            state: Functional state to record.
            step: Checkpoint step.
            metrics: Host scalar metrics.
            force: Whether policy bypass was requested.
            callback_states: Optional persistent callback state.

        Returns:
            Always `True`.
        """
        self.saved.append((state, step, dict(metrics or {}), force, dict(callback_states or {})))
        return True

    @property
    def steps(self) -> tuple[int, ...]:
        """Return saved checkpoint steps.

        Returns:
            Ordered recorded steps.
        """
        return tuple(record[1] for record in self.saved if record[1] not in self.deleted)

    def checkpoint_path(self, step: int) -> Path:
        """Return a deterministic mock checkpoint path.

        Args:
            step: Checkpoint step.

        Returns:
            Synthetic checkpoint path.
        """
        return Path(f"/checkpoints/{step}")

    def delete(self, step: int) -> None:
        """Record checkpoint deletion.

        Args:
            step: Checkpoint step.
        """
        self.deleted.append(step)

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

    def restore_callback_states(self, step: int | None = None) -> dict[str, Mapping[str, Any]]:
        """Return no persistent callback state for the recording backend.

        Args:
            step: Optional checkpoint step.

        Returns:
            Empty callback state mapping.
        """
        del step
        return {}

    def wait_until_finished(self) -> None:
        """Record that pending writes were awaited."""
        self.waited = True

    def close(self) -> None:
        """Record pending-write synchronization and backend closure."""
        self.wait_until_finished()
        self.closed = True
        self.opened = False


def test_model_checkpoint_saves_periodic_and_terminal_states(caplog: pytest.LogCaptureFixture) -> None:
    """Verify checkpoint scheduling and scalar metric transfer are callback-owned."""
    checkpoint_io = _RecordingCheckpointIO()
    callback = ModelCheckpoint(checkpoint_io, every_n_steps=2, save_last=True)
    with caplog.at_level(logging.INFO, logger="phijax.callbacks.model_checkpoint"):
        callback.setup()
        callback.on_fit_start(TrainerContext("initial", 0, {}))
        callback.on_train_metrics(TrainerContext("one", 1, {"loss": jnp.asarray(3.0)}))
        callback.on_train_metrics(TrainerContext("two", 2, {"loss": jnp.asarray(2.0)}))
        callback.on_fit_end(TrainerContext("three", 3, {"loss": jnp.asarray(1.0)}))
        callback.teardown()
        callback.close()

    assert checkpoint_io.saved == [
        ("two", 2, {"loss": 2.0}, True, {}),
        ("three", 3, {"loss": 1.0}, True, {}),
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
    callback = ModelCheckpoint(checkpoint_io, every_n_steps=1, monitor="loss")
    with pytest.raises(ValueError, match="scalar"):
        callback.on_train_metrics(TrainerContext(None, 1, {"loss": jnp.ones(2)}))


def test_model_checkpoint_optionally_saves_last_valid_state_on_exception() -> None:
    """Verify exceptional shutdown checkpointing is explicit and deduplicates the current step."""
    checkpoint_io = _RecordingCheckpointIO()
    callback = ModelCheckpoint(checkpoint_io, save_last=False, save_on_exception=True)
    context = TrainerContext("valid", 7, {"loss": jnp.asarray(0.5)})

    callback.setup()
    callback.on_fit_start(TrainerContext("initial", 0, {}))
    callback.on_exception(KeyboardInterrupt(), context)
    callback.on_exception(RuntimeError("duplicate"), context)

    assert checkpoint_io.saved == [("valid", 7, {"loss": 0.5}, True, {})]


def test_model_checkpoint_retains_best_monitored_states() -> None:
    """Verify monitored top-k retention, deterministic ties, and public paths."""
    checkpoint_io = _RecordingCheckpointIO()
    callback = ModelCheckpoint(
        checkpoint_io,
        every_n_steps=1,
        save_last=False,
        monitor="train/loss",
        mode="min",
        save_top_k=2,
    )
    callback.setup()
    callback.on_fit_start(TrainerContext(None, 0, {}))
    for step, loss in ((1, 3.0), (2, 2.0), (3, 2.0), (4, 1.0)):
        callback.on_train_metrics(TrainerContext(str(step), step, {"train/loss": jnp.asarray(loss)}))

    assert checkpoint_io.steps == (2, 4)
    assert checkpoint_io.deleted == [1, 3]
    assert callback.best_model_path == Path("/checkpoints/4")
    assert callback.best_model_score == 1.0


def test_model_checkpoint_rejects_missing_monitored_metric() -> None:
    """Verify a configured metric must be available at checkpoint time."""
    callback = ModelCheckpoint(
        _RecordingCheckpointIO(),
        every_n_steps=1,
        save_last=False,
        monitor="validation/score",
    )
    callback.setup()
    callback.on_fit_start(TrainerContext(None, 0, {}))
    with pytest.raises(ValueError, match=r"validation/score.*unavailable"):
        callback.on_train_metrics(TrainerContext(None, 1, {"train/loss": jnp.asarray(1.0)}))


def test_model_checkpoint_state_round_trip() -> None:
    """Verify checkpoint ranking state restores without implicit coercion."""
    callback = ModelCheckpoint(_RecordingCheckpointIO(), monitor="train/loss", save_last=False)
    state = {
        "scores": {"2": 0.25},
        "last_saved_step": 2,
        "best_model_path": "/checkpoints/2",
        "best_model_score": 0.25,
        "last_model_path": None,
    }
    callback.load_state_dict(state)

    assert callback.state_dict() == state
