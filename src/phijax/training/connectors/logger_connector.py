from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from phijax.callbacks import Callback, TrainerContext
from phijax.metrics import _LoggedMetric, _metric_is_scalar
from phijax.training.loggers import ExperimentLogger, LoggerCollection, create_default_logger


@dataclass(slots=True)
class _MetricStore:
    """Store the latest complete, logger, and progress metric views.

    Attributes:
        callback: Complete mapping available to callbacks and checkpoint monitors.
        logged: Scalar values selected for experiment loggers.
        progress_bar: Scalar values selected for progress displays.
    """

    callback: dict[str, Any]
    logged: dict[str, Any]
    progress_bar: dict[str, Any]


class _LoggerConnector:
    """Resolve loggers and own the Trainer's run-scoped metric views."""

    def __init__(
        self,
        logger: bool | ExperimentLogger | Iterable[ExperimentLogger] | None,
        default_root_dir: Path,
        *,
        callbacks: Sequence[Callback],
        is_global_zero: bool,
    ) -> None:
        """Initialize logger configuration and empty metric views.

        Args:
            logger: Default flag, one logger, several loggers, or `None`.
            default_root_dir: Parent directory for versioned local default logs.
            callbacks: Ordered callbacks that may contribute host-side metrics.
            is_global_zero: Whether this process owns default logger creation.
        """
        self._default_root_dir = default_root_dir
        self._callbacks = tuple(callbacks)
        self._is_global_zero = is_global_zero
        self._logger, self._has_logger = _resolve_loggers(
            logger,
            default_root_dir,
            is_global_zero=is_global_zero,
        )
        self._metrics = _MetricStore(callback={}, logged={}, progress_bar={})

    @property
    def logger(self) -> LoggerCollection:
        """Return the active logger collection.

        Returns:
            Configured experiment logger collection.
        """
        return self._logger

    @property
    def has_logger(self) -> bool:
        """Return whether at least one experiment logger was configured.

        Returns:
            Process-independent configured-logger flag.
        """
        return self._has_logger

    @property
    def callback_metrics(self) -> Mapping[str, Any]:
        """Return the complete metric mapping for callbacks and monitors.

        Returns:
            Latest complete metric mapping.
        """
        return self._metrics.callback

    @property
    def logged_metrics(self) -> Mapping[str, Any]:
        """Return metrics selected for experiment loggers.

        Returns:
            Latest logger metric mapping.
        """
        return self._metrics.logged

    @property
    def progress_bar_metrics(self) -> Mapping[str, Any]:
        """Return metrics selected for progress displays.

        Returns:
            Latest progress metric mapping.
        """
        return self._metrics.progress_bar

    def set_logger(self, logger: ExperimentLogger | Iterable[ExperimentLogger] | None) -> None:
        """Replace configured experiment loggers before running a task.

        Args:
            logger: One logger, several loggers, or `None` to disable logging.
        """
        self._logger, self._has_logger = _resolve_loggers(
            logger,
            self._default_root_dir,
            is_global_zero=self._is_global_zero,
        )

    def collect_callback_metrics(
        self,
        context: TrainerContext,
        existing_metrics: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Collect and validate metrics contributed by configured callbacks.

        Args:
            context: Current post-module Trainer context.
            existing_metrics: Metrics already produced by the compiled step and module.

        Returns:
            Uniquely named callback metrics in callback declaration order.

        Raises:
            TypeError: If a callback does not return a mapping.
            ValueError: If a callback returns an invalid or colliding metric name.
        """
        callback_metrics: dict[str, Any] = {}
        for callback in self._callbacks:
            contributed_metrics = callback.training_metrics(context)
            if not isinstance(contributed_metrics, Mapping):
                raise TypeError(f"{type(callback).__name__}.training_metrics() must return a mapping.")
            invalid_names = tuple(name for name in contributed_metrics if not isinstance(name, str) or not name.strip())
            if invalid_names:
                raise ValueError(f"Callback metric names must be non-empty strings: {invalid_names}.")
            existing_names = set(existing_metrics) | set(callback_metrics)
            collisions = existing_names & set(contributed_metrics)
            if collisions:
                raise ValueError(f"Callback metrics collide with existing names: {sorted(collisions)}.")
            callback_metrics.update(contributed_metrics)
        return callback_metrics

    def set_metrics(
        self,
        metrics: Mapping[str, Any],
        module_logs: Mapping[str, _LoggedMetric] | None = None,
        callback_names: Sequence[str] = (),
    ) -> None:
        """Update all metric views from one completed host-side logging phase.

        Args:
            metrics: Complete module and callback metric mapping.
            module_logs: Optional module declarations collected through :meth:`phijax.core.BasePhiModule.log`.
            callback_names: Metrics contributed by callbacks and logged by default when scalar.
        """
        callback = dict(metrics)
        records = {} if module_logs is None else module_logs
        logged = {
            name: record.value for name, record in records.items() if record.logger and _metric_is_scalar(record.value)
        }
        progress_bar = {
            name: record.value
            for name, record in records.items()
            if record.prog_bar and _metric_is_scalar(record.value)
        }
        for name in callback_names:
            value = callback[name]
            if _metric_is_scalar(value):
                logged[name] = value
        self._metrics = _MetricStore(callback=callback, logged=logged, progress_bar=progress_bar)


def _resolve_loggers(
    logger: bool | ExperimentLogger | Iterable[ExperimentLogger] | None,
    default_root_dir: Path,
    *,
    is_global_zero: bool,
) -> tuple[LoggerCollection, bool]:
    """Normalize Lightning-style logger configuration.

    Args:
        logger: Default flag, one logger, several loggers, or `None`.
        default_root_dir: Parent directory for versioned local default logs.
        is_global_zero: Whether this process owns default logger creation.

    Returns:
        Active logger collection and process-independent configured-logger flag.

    Raises:
        TypeError: If `logger` contains unsupported values.
    """
    if logger is True:
        return create_default_logger(default_root_dir, is_global_zero=is_global_zero), True
    if logger is False or logger is None:
        return LoggerCollection(), False
    if isinstance(logger, LoggerCollection):
        return logger, bool(logger)
    if isinstance(logger, ExperimentLogger):
        return LoggerCollection((logger,)), True
    if isinstance(logger, Iterable) and not isinstance(logger, (str, bytes)):
        loggers = tuple(logger)
        if any(not isinstance(candidate, ExperimentLogger) for candidate in loggers):
            raise TypeError("Every `logger` iterable entry must be an `ExperimentLogger`.")
        return LoggerCollection(loggers), bool(loggers)
    raise TypeError("`logger` must be a boolean, `None`, an `ExperimentLogger`, or an iterable of loggers.")
