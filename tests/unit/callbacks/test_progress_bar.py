import io
from pathlib import Path

import jax.numpy as jnp
import pytest

from phijax.callbacks import ModelSummary, ProgressBar, RichProgressBar, TQDMProgressBar, TrainerContext
from phijax.training import Trainer
from tests.helpers.progress import ProgressTrainer


def _train_step(
    callback: ProgressBar, trainer: ProgressTrainer, step: int, metrics: dict[str, object]
) -> TrainerContext:
    """Route one step's metrics through the fake Trainer and advance the display.

    Args:
        callback: Connected progress display.
        trainer: Fake Trainer that the display reads metrics from.
        step: Completed global step.
        metrics: Metrics produced by the step.

    Returns:
        Context passed to the display, for reuse in a terminal hook.
    """
    trainer.route(metrics)
    context = TrainerContext(state=None, step=step, metrics=metrics)
    callback.on_train_metrics(context)
    return context


def test_tqdm_progress_bar_displays_only_selected_metrics() -> None:
    """Verify TQDM renders one compact task with explicitly selected scalar metrics."""
    stream = io.StringIO()
    callback = TQDMProgressBar(refresh_rate=1, metric_names=("train/loss", "metric/missing"), stream=stream)
    trainer = ProgressTrainer(max_steps=2)
    callback.connect(trainer)
    callback.setup()
    callback.on_fit_start(TrainerContext(state=None, step=0, metrics={}))
    _train_step(callback, trainer, 1, {"train/loss": jnp.asarray(2.0)})
    callback.on_fit_end(_train_step(callback, trainer, 2, {"train/loss": jnp.asarray(1.0)}))

    rendered = " ".join(stream.getvalue().split())
    assert "Training" in rendered
    assert "2/2" in rendered
    assert "train/loss=1.000e+00" in rendered
    assert "metric/missing" not in rendered


def test_tqdm_progress_bar_resumes_from_restored_global_step() -> None:
    """Verify a resumed fit starts the bar at the restored step and refreshes metrics on its first update."""
    stream = io.StringIO()
    callback = TQDMProgressBar(refresh_rate=4, metric_names=("train/loss",), stream=stream)
    trainer = ProgressTrainer(max_steps=8)
    callback.connect(trainer)
    callback.setup()
    callback.on_fit_start(TrainerContext(state=None, step=5, metrics={}))
    assert callback._bar is not None
    bar = callback._bar
    assert bar.n == 5

    _train_step(callback, trainer, 6, {"train/loss": jnp.asarray(3.0)})
    assert "train/loss=3.000e+00" in " ".join(stream.getvalue().split())
    for step in (7, 8):
        final = _train_step(callback, trainer, step, {"train/loss": jnp.asarray(1.0)})
    assert bar.n == 8
    callback.on_fit_end(final)

    rendered = " ".join(stream.getvalue().split())
    assert "8/8" in rendered
    assert "9/8" not in rendered


def test_progress_bar_can_be_disabled_and_enabled() -> None:
    """Verify the public display switch suppresses and restores rendering."""
    stream = io.StringIO()
    callback = TQDMProgressBar(stream=stream)
    callback.connect(ProgressTrainer(max_steps=1))
    callback.disable()
    callback.setup()
    context = TrainerContext(state=None, step=0, metrics={})
    callback.on_fit_start(context)
    callback.on_fit_end(context)
    assert stream.getvalue() == ""

    callback.enable()
    callback.setup()
    callback.on_fit_start(context)
    callback.on_fit_end(context)
    assert "Training" in stream.getvalue()


def test_trainer_automatically_configures_default_displays(tmp_path: Path) -> None:
    """Verify enabled display flags add one TQDM bar and one plain summary."""
    trainer = Trainer(max_steps=2, logger=False, default_root_dir=tmp_path)

    assert sum(isinstance(callback, ProgressBar) for callback in trainer.callbacks) == 1
    assert any(isinstance(callback, TQDMProgressBar) for callback in trainer.callbacks)
    assert sum(isinstance(callback, ModelSummary) for callback in trainer.callbacks) == 1


def test_trainer_disables_configured_progress_bar() -> None:
    """Verify the Trainer flag disables an explicit Rich replacement through its public API."""
    progress = RichProgressBar()
    trainer = Trainer(
        max_steps=2,
        callbacks=(progress,),
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    assert trainer.callbacks == (progress,)
    assert progress.is_disabled


def test_trainer_rejects_multiple_progress_bars() -> None:
    """Verify one task cannot own competing terminal progress displays."""
    callbacks = (TQDMProgressBar(), RichProgressBar())
    with pytest.raises(ValueError, match="Only one progress-bar"):
        Trainer(max_steps=2, callbacks=callbacks, logger=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"refresh_rate": 0}, "`refresh_rate` must be positive"),
        ({"description": " "}, "`description` must not be empty"),
        ({"predict_description": " "}, "`predict_description` must not be empty"),
        ({"metric_names": ("loss", "loss")}, "unique non-empty"),
    ],
)
def test_tqdm_progress_bar_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    """Verify invalid progress options fail during construction.

    Args:
        kwargs: Constructor options under test.
        message: Expected validation error fragment.
    """
    with pytest.raises(ValueError, match=message):
        TQDMProgressBar(**kwargs)  # type: ignore[arg-type]


def test_tqdm_progress_bar_rejects_non_scalar_metrics() -> None:
    """Verify routed progress values must be scalar."""
    callback = TQDMProgressBar(stream=io.StringIO())
    trainer = ProgressTrainer(max_steps=1)
    callback.connect(trainer)
    callback.setup()
    callback.on_fit_start(TrainerContext(state=None, step=0, metrics={}))
    with pytest.raises(ValueError, match="must be scalar"):
        _train_step(callback, trainer, 1, {"train/loss": jnp.asarray([1.0, 2.0])})


@pytest.mark.parametrize("progress_type", [TQDMProgressBar, RichProgressBar])
def test_progress_bar_total_follows_trainer_max_steps(progress_type: type[ProgressBar]) -> None:
    """Verify a configured display counts to the owning Trainer's `max_steps` without its own total.

    Args:
        progress_type: Progress display class under test.
    """
    progress = progress_type()
    Trainer(max_steps=7, callbacks=(progress,), enable_model_summary=False, logger=False)

    assert progress.total == 7


def test_unconnected_progress_bar_rejects_fit() -> None:
    """Verify an enabled display without a Trainer reports the missing connection when fitting starts."""
    callback = TQDMProgressBar(stream=io.StringIO())
    callback.setup()

    with pytest.raises(RuntimeError, match="must be connected to a `Trainer`"):
        callback.on_fit_start(TrainerContext(state=None, step=0, metrics={}))
