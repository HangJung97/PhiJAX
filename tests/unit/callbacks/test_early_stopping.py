import json
import logging
from types import SimpleNamespace
from typing import Any

import jax.numpy as jnp
import pytest

from phijax.callbacks import EarlyStopping, TrainerContext


def _connected(**kwargs: Any) -> tuple[EarlyStopping, SimpleNamespace]:
    """Create an early-stopping callback connected to a minimal Trainer stand-in.

    Args:
        **kwargs: `EarlyStopping` options.

    Returns:
        Callback and the object whose `should_stop` flag it sets.
    """
    trainer = SimpleNamespace(should_stop=False)
    callback = EarlyStopping(**kwargs)
    callback.connect(trainer)
    return callback, trainer


def _check(callback: EarlyStopping, step: int, value: float) -> None:
    """Run one metric check at a global step.

    Args:
        callback: Early-stopping callback under test.
        step: Global optimizer step of the check.
        value: Monitored loss value.
    """
    callback.on_train_metrics(TrainerContext(None, step, {"loss": jnp.asarray(value)}))


def test_early_stopping_stops_once_wait_count_reaches_patience() -> None:
    """Verify Lightning's patience rule: stop once `wait_count >= patience`, and improvements reset the count."""
    callback, trainer = _connected(monitor="loss", patience=2, min_delta=0.1)

    _check(callback, 1, 2.0)
    _check(callback, 2, 1.95)
    assert (callback.best_score, callback.wait_count, trainer.should_stop) == (2.0, 1, False)
    _check(callback, 3, 1.5)
    assert (callback.best_score, callback.wait_count, trainer.should_stop) == (1.5, 0, False)
    _check(callback, 4, 1.5)
    _check(callback, 5, 1.49)

    assert trainer.should_stop is True
    assert callback.wait_count == 2
    assert callback.stopped_step == 5


def test_early_stopping_supports_max_mode() -> None:
    """Verify larger values count as improvements in `max` mode."""
    callback, trainer = _connected(monitor="loss", patience=1, mode="max")

    _check(callback, 1, 0.5)
    _check(callback, 2, 0.75)
    assert callback.best_score == 0.75
    _check(callback, 3, 0.625)

    assert trainer.should_stop is True


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_early_stopping_stops_on_non_finite_metrics(value: float, caplog: pytest.LogCaptureFixture) -> None:
    """Verify `check_finite` stops immediately, and disabling it treats the value as a non-improvement.

    Args:
        value: Non-finite monitored value.
        caplog: Pytest log capture fixture.
    """
    callback, trainer = _connected(monitor="loss", patience=5)
    _check(callback, 1, 1.0)
    with caplog.at_level(logging.INFO, logger="phijax.callbacks.early_stopping"):
        _check(callback, 2, value)

    assert trainer.should_stop is True
    assert callback.stopped_step == 2
    assert "is not finite" in caplog.text

    unchecked, unchecked_trainer = _connected(monitor="loss", patience=5, check_finite=False)
    _check(unchecked, 1, 1.0)
    _check(unchecked, 2, value)
    assert (unchecked.best_score, unchecked.wait_count, unchecked_trainer.should_stop) == (1.0, 1, False)


def test_early_stopping_checks_only_every_n_global_steps() -> None:
    """Verify skipped steps neither read the metric nor change progress."""
    callback, trainer = _connected(monitor="loss", patience=1, check_every_n_steps=3)

    # Skipped steps do not require the monitored metric.
    callback.on_train_metrics(TrainerContext(None, 1, {}))
    _check(callback, 2, 0.0)
    assert callback.best_score is None
    _check(callback, 3, 1.0)
    _check(callback, 4, 2.0)
    _check(callback, 5, 2.0)
    assert (callback.best_score, callback.wait_count, trainer.should_stop) == (1.0, 0, False)
    _check(callback, 6, 2.0)

    assert trainer.should_stop is True
    assert callback.stopped_step == 6


def test_early_stopping_state_round_trip_continues_like_an_uninterrupted_run() -> None:
    """Verify restored progress stops at the same step as an uninterrupted run."""
    values = [3.0, 2.0, 2.5, 2.4, 2.3]
    uninterrupted, uninterrupted_trainer = _connected(monitor="loss", patience=3)
    for step, value in enumerate(values, start=1):
        _check(uninterrupted, step, value)

    first, _ = _connected(monitor="loss", patience=3)
    for step, value in enumerate(values[:3], start=1):
        _check(first, step, value)
    state = json.loads(json.dumps(first.state_dict()))
    resumed, resumed_trainer = _connected(monitor="loss", patience=3)
    resumed.load_state_dict(state)
    for step, value in enumerate(values[3:], start=4):
        _check(resumed, step, value)

    settings = {"monitor": "loss", "mode": "min", "patience": 3}
    assert state == {"wait_count": 1, "best_score": 2.0, "stopped_step": 0, **settings}
    assert uninterrupted_trainer.should_stop is resumed_trainer.should_stop is True
    assert resumed.state_dict() == uninterrupted.state_dict()
    assert resumed.state_dict() == {"wait_count": 3, "best_score": 2.0, "stopped_step": 5, **settings}


@pytest.mark.parametrize(
    ("stored", "restored"),
    [({"monitor": "other"}, False), ({"mode": "max"}, False), ({"patience": 7}, True)],
    ids=["monitor", "mode", "patience"],
)
def test_early_stopping_load_checks_stored_settings(stored: dict[str, Any], restored: bool) -> None:
    """Verify progress for another metric or mode is skipped, and a changed patience keeps the configured value.

    Args:
        stored: Stored setting that differs from the callback's configuration.
        restored: Whether the stored progress is expected to be restored.
    """
    callback = EarlyStopping(monitor="loss", patience=3)
    state = {"wait_count": 2, "best_score": 1.0, "stopped_step": 0, "monitor": "loss", "mode": "min", "patience": 3}

    with pytest.warns(UserWarning, match=r"skipping stored progress|configured patience=3"):
        callback.load_state_dict({**state, **stored})

    assert callback.patience == 3
    assert (callback.wait_count, callback.best_score) == ((2, 1.0) if restored else (0, None))


def test_early_stopping_restores_progress_stored_without_settings() -> None:
    """Verify progress written before settings were stored still restores."""
    callback = EarlyStopping(monitor="loss")

    callback.load_state_dict({"wait_count": 2, "best_score": 1.0, "stopped_step": 0})

    assert (callback.wait_count, callback.best_score) == (2, 1.0)


def test_early_stopping_rejects_missing_or_non_scalar_metrics_and_invalid_options() -> None:
    """Verify monitored values must exist and be scalar, and the check interval must be positive."""
    callback, _ = _connected(monitor="loss")
    with pytest.raises(KeyError, match="loss"):
        callback.on_train_metrics(TrainerContext(None, 1, {}))
    with pytest.raises(ValueError, match="scalar"):
        callback.on_train_metrics(TrainerContext(None, 1, {"loss": jnp.ones(2)}))
    for interval in (0, True, 1.5):
        with pytest.raises(ValueError, match="check_every_n_steps"):
            EarlyStopping(check_every_n_steps=interval)  # type: ignore[arg-type]
