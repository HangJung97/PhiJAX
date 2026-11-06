from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

import jax
import numpy as np


@jax.tree_util.register_dataclass
@dataclass(frozen=True, slots=True)
class TrainingOutput:
    """Return named objective losses and optional compiled diagnostics.

    Attributes:
        losses: Stable scalar losses consumed by the configured loss balancer.
        diagnostics: Stable auxiliary arrays returned to the host without affecting gradients.
    """

    losses: Mapping[str, jax.Array]
    diagnostics: Mapping[str, jax.Array] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class _LoggedMetric:
    """Store one host-side metric value and its destinations.

    Attributes:
        value: Host scalar or device value retained without conversion.
        logger: Whether experiment loggers receive the metric.
        prog_bar: Whether progress callbacks display the metric.
    """

    value: Any
    logger: bool
    prog_bar: bool


class _ModuleMetricCollector:
    """Collect module log declarations during one batch-end hook."""

    def __init__(self) -> None:
        """Initialize an empty ordered metric collection."""
        self.records: dict[str, _LoggedMetric] = {}

    def log(self, name: str, value: Any, *, logger: bool, prog_bar: bool) -> None:
        """Record an explicit module metric declaration.

        Args:
            name: Complete metric name.
            value: Host scalar or device value.
            logger: Whether experiment loggers receive the value.
            prog_bar: Whether progress callbacks display the value.
        """
        self.records[name] = _LoggedMetric(value, logger, prog_bar)

    def add_defaults(self, metrics: Mapping[str, Any]) -> None:
        """Add default destinations for scalar metrics not explicitly declared.

        Args:
            metrics: Complete metric mapping returned by the module hook.
        """
        for name, value in metrics.items():
            if name not in self.records and _metric_is_scalar(value):
                in_progress = name == "train/loss" or name.startswith(("train/loss/", "train/weight/"))
                self.records[name] = _LoggedMetric(value, logger=True, prog_bar=in_progress)


_ACTIVE_MODULE_METRICS: ContextVar[_ModuleMetricCollector | None] = ContextVar(
    "phijax_active_module_metrics",
    default=None,
)


@contextmanager
def _collect_module_metrics() -> Iterator[_ModuleMetricCollector]:
    """Activate an isolated metric collector for one module batch-end hook.

    Yields:
        Empty collector receiving calls to :meth:`phijax.core.BasePhiModule.log`.
    """
    collector = _ModuleMetricCollector()
    token = _ACTIVE_MODULE_METRICS.set(collector)
    try:
        yield collector
    finally:
        _ACTIVE_MODULE_METRICS.reset(token)


def _metric_is_scalar(value: Any) -> bool:
    """Return whether a metric has exactly one element without transferring JAX arrays.

    Args:
        value: Host scalar or array-like metric value.

    Returns:
        Whether scalar loggers and progress displays can consume the value.
    """
    size = getattr(value, "size", None)
    return np.asarray(value).size == 1 if size is None else int(size) == 1


__all__ = ["TrainingOutput"]
