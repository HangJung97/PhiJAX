import io
from pathlib import Path

import jax.numpy as jnp
import pytest

from phijax.callbacks import ModelSummary, ProgressBar, RichProgressBar, TQDMProgressBar, TrainerContext
from phijax.training import Trainer


def test_tqdm_progress_bar_displays_only_selected_metrics() -> None:
    """Verify TQDM renders one compact task with explicitly selected scalar metrics."""
    stream = io.StringIO()
    callback = TQDMProgressBar(
        total=2,
        refresh_rate=1,
        metric_names=("train/loss", "metric/missing"),
        stream=stream,
    )
    callback.setup()
    callback.on_fit_start(TrainerContext(state=None, step=0, metrics={}))
    callback.on_train_metrics(TrainerContext(state=None, step=1, metrics={"train/loss": jnp.asarray(2.0)}))
    final = TrainerContext(state=None, step=2, metrics={"train/loss": jnp.asarray(1.0)})
    callback.on_train_metrics(final)
    callback.on_fit_end(final)

    rendered = " ".join(stream.getvalue().split())
    assert "Training" in rendered
    assert "2/2" in rendered
    assert "train/loss=1.000e+00" in rendered
    assert "metric/missing" not in rendered


def test_progress_bar_can_be_disabled_and_enabled() -> None:
    """Verify the public display switch suppresses and restores rendering."""
    stream = io.StringIO()
    callback = TQDMProgressBar(total=1, stream=stream)
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
    progress = RichProgressBar(total=2)
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
    callbacks = (TQDMProgressBar(total=2), RichProgressBar(total=2))
    with pytest.raises(ValueError, match="Only one progress-bar"):
        Trainer(max_steps=2, callbacks=callbacks, logger=False)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"total": 0}, "`total` must be positive"),
        ({"total": 1, "refresh_rate": 0}, "`refresh_rate` must be positive"),
        ({"total": 1, "description": " "}, "`description` must not be empty"),
        ({"total": 1, "predict_description": " "}, "`predict_description` must not be empty"),
        ({"total": 1, "metric_names": ("loss", "loss")}, "unique non-empty"),
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
    callback = TQDMProgressBar(total=1, stream=io.StringIO())
    callback.setup()
    callback.on_fit_start(TrainerContext(state=None, step=0, metrics={}))
    with pytest.raises(ValueError, match="must be scalar"):
        callback.on_train_metrics(TrainerContext(state=None, step=1, metrics={"train/loss": jnp.asarray([1.0, 2.0])}))
