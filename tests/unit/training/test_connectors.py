from pathlib import Path
from typing import Any

import jax.numpy as jnp
import pytest

from phijax.callbacks import Callback, TrainerContext
from phijax.metrics import _LoggedMetric
from phijax.training.connectors.logger_connector import _LoggerConnector


class _MetricCallback(Callback):
    """Contribute one host-routed callback metric."""

    def training_metrics(self, context: TrainerContext) -> dict[str, Any]:
        """Return one scalar derived from the completed step.

        Args:
            context: Completed-step Trainer context.

        Returns:
            One uniquely named callback metric.
        """
        return {"callback/step": jnp.asarray(context.step)}


def test_logger_connector_owns_metric_views_and_callback_collection(tmp_path: Path) -> None:
    """Verify the connector separates callback, logger, and progress metric views.

    Args:
        tmp_path: Temporary default logger root.
    """
    connector = _LoggerConnector(False, tmp_path, callbacks=(_MetricCallback(),), is_global_zero=True)
    context = TrainerContext(state=None, step=3, metrics={"train/loss": jnp.asarray(2.0)})

    callback_metrics = connector.collect_callback_metrics(context, context.metrics)
    metrics = {**context.metrics, **callback_metrics, "diagnostic/vector": jnp.ones(2)}
    connector.set_metrics(
        metrics,
        {"train/loss": _LoggedMetric(jnp.asarray(2.0), logger=True, prog_bar=True)},
        tuple(callback_metrics),
    )

    assert set(connector.callback_metrics) == {"train/loss", "callback/step", "diagnostic/vector"}
    assert set(connector.logged_metrics) == {"train/loss", "callback/step"}
    assert set(connector.progress_bar_metrics) == {"train/loss"}


def test_logger_connector_rejects_callback_metric_collisions(tmp_path: Path) -> None:
    """Verify callback metrics cannot replace compiled or module metrics.

    Args:
        tmp_path: Temporary default logger root.
    """
    connector = _LoggerConnector(False, tmp_path, callbacks=(_MetricCallback(),), is_global_zero=True)
    context = TrainerContext(state=None, step=1, metrics={})

    with pytest.raises(ValueError, match="collide"):
        connector.collect_callback_metrics(context, {"callback/step": jnp.asarray(0)})
