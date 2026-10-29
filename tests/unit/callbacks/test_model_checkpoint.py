import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from phijax.balancers import BalancerState
from phijax.callbacks import ModelCheckpoint, TrainerContext
from phijax.training import OrbaxCheckpointIO, TrainState


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
    callback.teardown()

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
        "last_step": 1,
        "best_model_path": "/checkpoints/2",
        "best_model_score": 0.25,
        "last_model_path": "/checkpoints/1",
    }
    callback.load_state_dict(state)

    assert callback.state_dict() == state


def test_model_checkpoint_loads_state_without_last_step() -> None:
    """Verify callback state saved before `last_step` existed still restores."""
    callback = ModelCheckpoint(_RecordingCheckpointIO())
    callback.load_state_dict(
        {
            "scores": {"2": 2.0},
            "last_saved_step": 2,
            "best_model_path": "/checkpoints/2",
            "best_model_score": None,
            "last_model_path": None,
        }
    )

    assert callback.state_dict()["last_step"] is None
    with pytest.raises(ValueError, match="last_step"):
        callback.load_state_dict({**callback.state_dict(), "last_step": "3"})


def _train_state(step: int) -> TrainState:
    """Build a small complete state accepted by the Orbax backend.

    Args:
        step: Optimizer step stored in the state.

    Returns:
        Complete functional training state.
    """
    return TrainState(
        model_state={"weight": jnp.asarray(float(step))},
        optimizer_state={"slot": jnp.asarray(0.0)},
        balancer_state=BalancerState(weights=jnp.ones(1), traces=jnp.zeros(1)),
        rng_key=jax.random.key(0),
        sampling_key=jax.random.key(1),
        balancer_key=jax.random.key(2),
        step=jnp.asarray(step, jnp.int32),
        loss_scale=jnp.asarray(1.0, jnp.float32),
        finite_steps=jnp.asarray(0, jnp.int32),
    )


def _run_steps(callback: ModelCheckpoint, steps: range, losses: dict[int, float] | None = None) -> None:
    """Drive one fit through periodic metric hooks and the fit-end hook.

    Args:
        callback: Checkpoint callback under test.
        steps: Completed training steps, in order.
        losses: Optional `loss` metric per step. Defaults to a decreasing loss.
    """
    callback.setup()
    callback.on_fit_start(TrainerContext(None, steps.start - 1, {}))
    for step in steps:
        loss = 10.0 - step if losses is None else losses[step]
        callback.on_train_metrics(TrainerContext(_train_state(step), step, {"loss": jnp.asarray(loss)}))
    last = steps[-1]
    callback.on_fit_end(TrainerContext(_train_state(last), last, {"loss": jnp.asarray(0.0)}))
    callback.teardown()


@pytest.mark.parametrize("asynchronous", [False, True])
def test_model_checkpoint_top_k_matches_orbax_storage(tmp_path: Path, asynchronous: bool) -> None:
    """Verify monitored top-k retention stays consistent with the steps Orbax stores.

    Args:
        tmp_path: Temporary checkpoint root.
        asynchronous: Whether Orbax writes checkpoints in a background thread.
    """
    checkpoint_io = OrbaxCheckpointIO(tmp_path / "checkpoints", enable_async_checkpointing=asynchronous)
    callback = ModelCheckpoint(checkpoint_io, every_n_steps=1, save_last=False, monitor="loss", save_top_k=5)
    _run_steps(callback, range(1, 9))

    assert checkpoint_io.steps == (4, 5, 6, 7, 8)
    assert sorted(callback.state_dict()["scores"], key=int) == ["4", "5", "6", "7", "8"]
    assert callback.best_model_path == checkpoint_io.checkpoint_path(8)
    assert callback.best_model_path is not None
    assert callback.best_model_path.exists()


def test_model_checkpoint_keep_all_monitored_retains_every_step(tmp_path: Path) -> None:
    """Verify `save_top_k=-1` keeps every monitored checkpoint."""
    checkpoint_io = OrbaxCheckpointIO(tmp_path / "checkpoints", enable_async_checkpointing=False)
    callback = ModelCheckpoint(checkpoint_io, every_n_steps=1, save_last=False, monitor="loss", save_top_k=-1)
    _run_steps(callback, range(1, 6), losses={step: float(step) for step in range(1, 6)})

    assert checkpoint_io.steps == (1, 2, 3, 4, 5)
    assert callback.best_model_path == checkpoint_io.checkpoint_path(1)


def test_model_checkpoint_default_keeps_only_latest_checkpoint(tmp_path: Path) -> None:
    """Verify the Lightning default `save_top_k=1` without `monitor` keeps the latest checkpoint."""
    checkpoint_io = OrbaxCheckpointIO(tmp_path / "checkpoints", enable_async_checkpointing=False)
    callback = ModelCheckpoint(checkpoint_io, every_n_steps=2)
    _run_steps(callback, range(1, 6))

    assert callback.save_top_k == 1
    assert checkpoint_io.steps == (5,)
    assert callback.best_model_path == checkpoint_io.checkpoint_path(5)
    assert callback.best_model_score is None
    assert callback.last_model_path == checkpoint_io.checkpoint_path(5)


def test_model_checkpoint_step_monitor_keeps_latest_k_checkpoints(tmp_path: Path) -> None:
    """Verify `monitor="step", mode="max"` keeps the latest `save_top_k` checkpoints."""
    checkpoint_io = OrbaxCheckpointIO(tmp_path / "checkpoints", enable_async_checkpointing=False)
    callback = ModelCheckpoint(
        checkpoint_io, every_n_steps=2, save_last=False, monitor="step", mode="max", save_top_k=3
    )
    _run_steps(callback, range(1, 11))

    assert checkpoint_io.steps == (6, 8, 10)
    assert callback.best_model_score == 10.0


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"save_top_k": 3}, r'monitor="step", mode="max"'),
        ({"save_top_k": -2}, "nonnegative"),
        ({"save_top_k": True}, "nonnegative"),
    ],
)
def test_model_checkpoint_validates_save_top_k(kwargs: dict[str, Any], match: str) -> None:
    """Verify `save_top_k` follows Lightning's rules with and without `monitor`.

    Args:
        kwargs: Invalid constructor options.
        match: Expected error-message pattern.
    """
    with pytest.raises(ValueError, match=match):
        ModelCheckpoint(_RecordingCheckpointIO(), **kwargs)
    for valid_top_k in (-1, 0, 1):
        ModelCheckpoint(_RecordingCheckpointIO(), save_top_k=valid_top_k)


def test_orbax_delete_ignores_missing_steps(tmp_path: Path) -> None:
    """Verify deleting a step that is not stored is a no-op."""
    with OrbaxCheckpointIO(tmp_path / "checkpoints", enable_async_checkpointing=False) as checkpoint_io:
        checkpoint_io.save(_train_state(1), 1)
        checkpoint_io.delete(7)
        checkpoint_io.delete(1)
        checkpoint_io.delete(1)

        assert checkpoint_io.steps == ()


def test_model_checkpoint_defers_deletion_until_next_write_commits() -> None:
    """Verify a displaced checkpoint stays stored until a later write has committed."""
    checkpoint_io = _RecordingCheckpointIO()
    callback = ModelCheckpoint(checkpoint_io, every_n_steps=1, save_last=False)
    callback.setup()
    callback.on_fit_start(TrainerContext(None, 0, {}))
    callback.on_train_metrics(TrainerContext("one", 1, {}))
    callback.on_train_metrics(TrainerContext("two", 2, {}))

    assert checkpoint_io.deleted == []
    assert checkpoint_io.waited is False

    callback.on_train_metrics(TrainerContext("three", 3, {}))

    assert checkpoint_io.waited is True
    assert checkpoint_io.deleted == [1]

    callback.teardown()

    assert checkpoint_io.deleted == [1, 2]


def test_model_checkpoint_keeps_last_checkpoint_when_displaced() -> None:
    """Verify top-k displacement removes the terminal checkpoint from ranking but not from storage."""
    checkpoint_io = _RecordingCheckpointIO()
    callback = ModelCheckpoint(checkpoint_io, every_n_steps=1, monitor="loss", save_top_k=1, save_last=True)
    callback.setup()
    callback.on_fit_start(TrainerContext(None, 0, {}))
    callback.on_fit_end(TrainerContext("last", 3, {"loss": jnp.asarray(2.0)}))
    callback.on_train_metrics(TrainerContext("better", 4, {"loss": jnp.asarray(1.0)}))
    callback.teardown()

    assert checkpoint_io.deleted == []
    assert callback.last_model_path == Path("/checkpoints/3")
    assert callback.best_model_path == Path("/checkpoints/4")
    assert callback.state_dict()["scores"] == {"4": 1.0}


def test_model_checkpoint_replaces_previous_last_checkpoint_across_fits(tmp_path: Path) -> None:
    """Verify a resumed fit replaces the previous terminal checkpoint instead of leaving it behind.

    Args:
        tmp_path: Temporary checkpoint root.
    """
    checkpoint_io = OrbaxCheckpointIO(tmp_path / "checkpoints", enable_async_checkpointing=False)
    callback = ModelCheckpoint(checkpoint_io, every_n_steps=4)
    _run_steps(callback, range(1, 7))
    first_fit_state = callback.state_dict()

    assert checkpoint_io.steps == (6,)

    callback.setup()
    callback.load_state_dict(first_fit_state)
    callback.on_fit_start(TrainerContext(None, 6, {}))
    for step in range(7, 10):
        callback.on_train_metrics(TrainerContext(_train_state(step), step, {}))
    callback.on_fit_end(TrainerContext(_train_state(9), 9, {}))
    callback.teardown()

    assert checkpoint_io.steps == (9,)
    assert callback.last_model_path == checkpoint_io.checkpoint_path(9)


def test_model_checkpoint_promotes_periodic_final_step_to_last() -> None:
    """Verify a final step already saved periodically becomes the terminal checkpoint."""
    checkpoint_io = _RecordingCheckpointIO()
    callback = ModelCheckpoint(checkpoint_io, every_n_steps=2, monitor="loss", save_top_k=1)
    callback.setup()
    callback.on_fit_start(TrainerContext(None, 0, {}))
    callback.on_train_metrics(TrainerContext("two", 2, {"loss": jnp.asarray(1.0)}))
    callback.on_fit_end(TrainerContext("two", 2, {"loss": jnp.asarray(1.0)}))
    callback.on_train_metrics(TrainerContext("four", 4, {"loss": jnp.asarray(0.5)}))
    callback.teardown()

    assert callback.last_model_path == Path("/checkpoints/2")
    assert checkpoint_io.deleted == []
    assert [record[1] for record in checkpoint_io.saved] == [2, 4]


def test_model_checkpoint_drops_ranked_steps_missing_from_storage() -> None:
    """Verify restored ranking state ignores checkpoints deleted outside the callback."""
    checkpoint_io = _RecordingCheckpointIO()
    checkpoint_io.save("kept", 2)
    callback = ModelCheckpoint(checkpoint_io, every_n_steps=1, monitor="loss", save_top_k=2, save_last=False)
    callback.load_state_dict(
        {
            "scores": {"1": 0.5, "2": 0.75},
            "last_saved_step": 2,
            "last_step": 1,
            "best_model_path": "/checkpoints/1",
            "best_model_score": 0.5,
            "last_model_path": "/checkpoints/1",
        }
    )
    callback.on_fit_start(TrainerContext(None, 2, {}))

    assert callback.state_dict()["scores"] == {"2": 0.75}
    assert callback.state_dict()["last_step"] is None
    assert callback.last_model_path is None
    assert callback.best_model_path == Path("/checkpoints/2")
    assert callback.best_model_score == 0.75
