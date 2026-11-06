import io

import jax.numpy as jnp
import pytest
from rich.console import Console

from phijax.callbacks import PredictionContext, RichProgressBar, TrainerContext
from tests.helpers.progress import ProgressTrainer


def _console(stream: io.StringIO) -> Console:
    """Create a deterministic non-interactive Rich console.

    Args:
        stream: Text stream receiving rendered progress output.

    Returns:
        Fixed-width console suitable for progress callback tests.
    """
    return Console(file=stream, force_terminal=False, width=120)


def test_rich_progress_bar_displays_selected_loss_and_metric() -> None:
    """Verify the callback renders progress plus selected available scalar metrics."""
    stream = io.StringIO()
    callback = RichProgressBar(
        refresh_rate=2,
        metric_names=("train/loss", "validation/relative_error", "metric/missing"),
        console=_console(stream),
    )
    trainer = ProgressTrainer(max_steps=2)
    callback.connect(trainer)
    first = {"train/loss": jnp.asarray(2.5), "validation/relative_error": jnp.asarray(0.125)}
    final = {"train/loss": jnp.asarray(1.25), "validation/relative_error": jnp.asarray(0.0625)}

    callback.setup()
    callback.on_fit_start(TrainerContext(state=None, step=0, metrics={}))
    trainer.route(first)
    callback.on_train_metrics(TrainerContext(state=None, step=1, metrics=first))
    trainer.route(final)
    final_context = TrainerContext(state=None, step=2, metrics=final)
    callback.on_train_metrics(final_context)
    callback.on_fit_end(final_context)
    callback.teardown()

    rendered = " ".join(stream.getvalue().split())
    assert "Training 2/2" in rendered
    assert "2/2" in rendered
    assert "it/s" in rendered
    assert "•" in rendered
    assert "train/loss: 1.250e+00" in rendered
    assert "validation/relative_error: 6.250e-02" in rendered
    assert "metric/missing" not in rendered


def test_rich_progress_bar_refreshes_device_metrics_at_configured_interval(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify intermediate batches advance without copying selected metrics to the host.

    Args:
        monkeypatch: Pytest fixture used to observe device transfers.
    """
    import phijax.callbacks.progress_bar as progress_module

    transfers: list[object] = []

    def record_device_get(value: object) -> object:
        """Record and preserve one selected metric value.

        Args:
            value: Metric value requested by the callback.

        Returns:
            The unchanged metric value.
        """
        transfers.append(value)
        return value

    monkeypatch.setattr(progress_module.jax, "device_get", record_device_get)
    callback = RichProgressBar(refresh_rate=3, console=_console(io.StringIO()))
    trainer = ProgressTrainer(max_steps=3)
    callback.connect(trainer)
    callback.setup()
    callback.on_fit_start(TrainerContext(state=None, step=0, metrics={}))

    for step in range(1, 4):
        metrics = {"train/loss": jnp.asarray(float(step))}
        trainer.route(metrics)
        callback.on_train_metrics(TrainerContext(state=None, step=step, metrics=metrics))

    assert len(transfers) == 2
    callback.teardown()


def test_rich_progress_bar_defaults_to_trainer_progress_metrics() -> None:
    """Verify default selection shows the Trainer's progress-bar metrics, not every routed metric."""
    stream = io.StringIO()
    callback = RichProgressBar(console=_console(stream))
    metrics = {
        "train/loss": jnp.asarray(3.0),
        "train/loss/fidelity/data": jnp.asarray(2.0),
        "train/loss/boundary/no_slip": jnp.asarray(1.0),
        "train/weight/fidelity/data": jnp.asarray(0.5),
        "train/precision/loss_scale": jnp.asarray(1.0),
    }
    trainer = ProgressTrainer(
        max_steps=1,
        progress_bar_metrics={"train/loss": metrics["train/loss"]},
        callback_metrics=metrics,
    )
    callback.connect(trainer)
    context = TrainerContext(state=None, step=1, metrics=metrics)

    callback.setup()
    callback.on_fit_start(TrainerContext(state=None, step=0, metrics={}))
    callback.on_train_metrics(context)
    callback.on_fit_end(context)

    rendered = " ".join(stream.getvalue().split())
    assert "train/loss: 3.000e+00" in rendered
    assert "train/loss/fidelity/data" not in rendered
    assert "train/loss/boundary/no_slip" not in rendered
    assert "train/weight/fidelity/data" not in rendered
    assert "train/precision/" not in rendered


def test_rich_progress_bar_displays_prediction_batch_progress() -> None:
    """Verify prediction hooks render the finite source length without training metrics."""
    stream = io.StringIO()
    callback = RichProgressBar(predict_description="Predicting", console=_console(stream))
    initial = PredictionContext(outputs=None, batch_index=None, total_batches=2, metadata={})

    callback.setup()
    callback.on_predict_start(initial)
    for batch_index in range(2):
        callback.on_predict_batch_end(
            PredictionContext(outputs=jnp.ones((1, 1)), batch_index=batch_index, total_batches=2, metadata={})
        )
    callback.on_predict_end(PredictionContext(outputs=jnp.ones((2, 1)), batch_index=None, total_batches=2, metadata={}))
    callback.teardown()

    rendered = " ".join(stream.getvalue().split())
    assert "Predicting 2/2" in rendered
    assert "2/2" in rendered
    assert "it/s" in rendered


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"refresh_rate": 0}, "`refresh_rate` must be positive"),
        ({"description": " "}, "`description` must not be empty"),
        ({"predict_description": " "}, "`predict_description` must not be empty"),
        ({"metric_names": ("loss", "loss")}, "unique non-empty"),
    ],
)
def test_rich_progress_bar_rejects_invalid_configuration(kwargs: dict[str, object], message: str) -> None:
    """Verify invalid progress display options fail during construction.

    Args:
        kwargs: Constructor options under test.
        message: Expected validation error fragment.
    """
    with pytest.raises(ValueError, match=message):
        RichProgressBar(**kwargs)  # type: ignore[arg-type]


def test_rich_progress_bar_rejects_non_scalar_selected_metrics() -> None:
    """Verify selected metrics must be scalar before Rich formatting."""
    callback = RichProgressBar(console=_console(io.StringIO()))
    trainer = ProgressTrainer(max_steps=1)
    callback.connect(trainer)
    callback.setup()
    callback.on_fit_start(TrainerContext(state=None, step=0, metrics={}))

    metrics = {"train/loss": jnp.asarray([1.0, 2.0])}
    trainer.route(metrics)
    with pytest.raises(ValueError, match="must be scalar"):
        callback.on_train_metrics(TrainerContext(state=None, step=1, metrics=metrics))
    callback.teardown()


def test_rich_progress_bar_resumes_from_restored_global_step() -> None:
    """Verify a resumed fit starts the bar at the restored step and refreshes metrics on its first update."""
    stream = io.StringIO()
    callback = RichProgressBar(refresh_rate=4, metric_names=("train/loss",), console=_console(stream))
    trainer = ProgressTrainer(max_steps=8)
    callback.connect(trainer)
    callback.setup()
    callback.on_fit_start(TrainerContext(state=None, step=5, metrics={}))
    assert callback._progress is not None
    task = callback._progress.tasks[0]
    assert task.completed == 5

    for step, loss in ((6, 3.0), (7, 2.0), (8, 1.0)):
        metrics = {"train/loss": jnp.asarray(loss)}
        trainer.route(metrics)
        final = TrainerContext(state=None, step=step, metrics=metrics)
        callback.on_train_metrics(final)
        if step == 6:
            assert "train/loss: 3.000e+00" in callback._last_metrics
    assert task.completed == 8
    callback.on_fit_end(final)
    callback.teardown()

    rendered = " ".join(stream.getvalue().split())
    assert "Training 8/8" in rendered
