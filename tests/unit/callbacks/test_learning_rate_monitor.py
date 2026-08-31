import jax.numpy as jnp
import pytest

from phijax.callbacks import LearningRateMonitor, TrainerContext


def test_learning_rate_monitor_requires_configured_logger() -> None:
    """Verify fitting fails before metrics are evaluated when logging is disabled."""
    monitor = LearningRateMonitor(lambda step: step)
    with pytest.raises(RuntimeError, match="Cannot use `LearningRateMonitor` with a Trainer that has no logger"):
        monitor.on_fit_start(TrainerContext(state=None, step=0, metrics={}, has_logger=False))

    monitor.on_fit_start(TrainerContext(state=None, step=0, metrics={}, has_logger=True))


def test_learning_rate_monitor_reports_the_rate_used_by_each_completed_step() -> None:
    """Verify completed trainer steps map back to the optimizer's zero-based schedule count."""
    monitor = LearningRateMonitor(lambda step: 0.1 * 0.5**step, log_key_prefix="train/")

    monitor.setup()
    first = monitor.training_metrics(TrainerContext(state=None, step=1, metrics={}, should_log=True))
    third = monitor.training_metrics(TrainerContext(state=None, step=3, metrics={}, should_log=True))

    assert float(first["train/lr"]) == pytest.approx(0.1)
    assert float(third["train/lr"]) == pytest.approx(0.025)


def test_learning_rate_monitor_supports_custom_names() -> None:
    """Verify callers may choose a stable application-specific learning-rate metric name."""
    monitor = LearningRateMonitor(lambda step: jnp.asarray(step + 1.0), name="optimizer/lr")

    metrics = monitor.training_metrics(TrainerContext(state=None, step=2, metrics={}, should_log=True))

    assert tuple(metrics) == ("optimizer/lr",)
    assert float(metrics["optimizer/lr"]) == pytest.approx(2.0)


def test_learning_rate_monitor_logs_prefixed_momentum_and_weight_decay() -> None:
    """Verify Lightning-compatible flags, prefixes, and suffixes include configured Optax hyperparameters."""
    monitor = LearningRateMonitor(
        lambda step: 0.1 * 0.5**step,
        log_momentum=True,
        log_weight_decay=True,
        log_key_prefix="optim/",
        momentum=0.9,
        weight_decay=lambda step: 0.01 * 0.1**step,
    )

    metrics = monitor.training_metrics(TrainerContext(state=None, step=2, metrics={}, should_log=True))

    assert set(metrics) == {"optim/lr", "optim/lr-momentum", "optim/lr-weight_decay"}
    assert float(metrics["optim/lr"]) == pytest.approx(0.05)
    assert float(metrics["optim/lr-momentum"]) == pytest.approx(0.9)
    assert float(metrics["optim/lr-weight_decay"]) == pytest.approx(0.001)


@pytest.mark.parametrize(
    ("schedule", "name", "exception", "match"),
    [
        (1.0, "train/lr", TypeError, "must be callable"),
        (lambda step: step, "", ValueError, "non-empty"),
    ],
)
def test_learning_rate_monitor_rejects_invalid_configuration(
    schedule: object,
    name: str,
    exception: type[Exception],
    match: str,
) -> None:
    """Verify schedule and metric-name validation fails during callback construction.

    Args:
        schedule: Invalid schedule value.
        name: Candidate metric name.
        exception: Expected exception type.
        match: Expected message fragment.
    """
    with pytest.raises(exception, match=match):
        LearningRateMonitor(schedule, name=name)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "exception", "match"),
    [
        ({"log_momentum": 1}, TypeError, "log_momentum"),
        ({"log_weight_decay": 1}, TypeError, "log_weight_decay"),
        ({"log_key_prefix": 1}, TypeError, "log_key_prefix"),
        ({"logging_interval": "batch"}, ValueError, "logging_interval"),
        ({"momentum": [0.8, 0.9]}, ValueError, "momentum"),
        ({"weight_decay": object()}, TypeError, "weight_decay"),
    ],
)
def test_learning_rate_monitor_rejects_invalid_optional_arguments(
    kwargs: dict[str, object],
    exception: type[Exception],
    match: str,
) -> None:
    """Verify Lightning-compatible options and explicit Optax values are validated.

    Args:
        kwargs: Invalid callback keyword arguments.
        exception: Expected exception type.
        match: Expected message fragment.
    """
    with pytest.raises(exception, match=match):
        LearningRateMonitor(lambda step: step, **kwargs)  # type: ignore[arg-type]


def test_learning_rate_monitor_rejects_missing_or_vector_schedules() -> None:
    """Verify executable callbacks require a scalar-producing schedule."""
    missing = LearningRateMonitor(None)
    vector = LearningRateMonitor(lambda step: jnp.asarray([step, step + 1]))
    vector_momentum = LearningRateMonitor(
        lambda step: step,
        log_momentum=True,
        momentum=lambda step: jnp.asarray([step, step + 1]),
    )

    with pytest.raises(RuntimeError, match="requires a configured"):
        missing.setup()
    with pytest.raises(ValueError, match="must return a scalar"):
        vector.training_metrics(TrainerContext(state=None, step=1, metrics={}, should_log=True))
    with pytest.raises(ValueError, match="must return a scalar"):
        vector_momentum.training_metrics(TrainerContext(state=None, step=1, metrics={}, should_log=True))


def test_learning_rate_monitor_respects_logging_interval() -> None:
    """Verify logger, step, and fit-end cadence choices avoid unnecessary schedule evaluation."""
    trainer_cadence = LearningRateMonitor(lambda step: step)
    step_cadence = LearningRateMonitor(lambda step: step, logging_interval="step")
    epoch_cadence = LearningRateMonitor(lambda step: step, logging_interval="epoch")
    ordinary = TrainerContext(state=None, step=2, metrics={})
    logging = TrainerContext(state=None, step=3, metrics={}, should_log=True)
    terminal = TrainerContext(state=None, step=4, metrics={}, is_fit_end=True)

    assert trainer_cadence.training_metrics(ordinary) == {}
    assert float(trainer_cadence.training_metrics(logging)["lr"]) == pytest.approx(2.0)
    assert float(trainer_cadence.training_metrics(terminal)["lr"]) == pytest.approx(3.0)
    assert float(step_cadence.training_metrics(ordinary)["lr"]) == pytest.approx(1.0)
    assert epoch_cadence.training_metrics(logging) == {}
    assert float(epoch_cadence.training_metrics(terminal)["lr"]) == pytest.approx(3.0)
