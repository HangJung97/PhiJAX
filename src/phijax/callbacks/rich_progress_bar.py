from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import jax
import numpy as np
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    Task,
    TaskID,
    TextColumn,
)
from rich.text import Text

from phijax.callbacks.base import Callback, CallbackContext, PredictionContext, TrainerContext


@dataclass(frozen=True, slots=True)
class RichProgressBarTheme:
    """Configure Lightning-style colors and formatting for :class:`RichProgressBar`.

    Attributes:
        description: Rich style applied to the step description.
        progress_bar: Rich style applied to the active portion of the bar.
        progress_bar_finished: Rich style applied to a completed bar.
        progress_bar_pulse: Rich style applied when the bar pulses.
        batch_progress: Rich style applied to the completed-step counter.
        time: Rich style applied to elapsed and remaining time.
        processing_speed: Rich style applied to iteration throughput.
        metrics: Rich style applied to displayed metrics.
        metrics_text_delimiter: Text separating adjacent metrics.
        metrics_format: Python format specification applied to scalar metrics.
    """

    description: str = ""
    progress_bar: str = "#6206E0"
    progress_bar_finished: str = "#6206E0"
    progress_bar_pulse: str = "#6206E0"
    batch_progress: str = ""
    time: str = "dim"
    processing_speed: str = "dim underline"
    metrics: str = "italic"
    metrics_text_delimiter: str = " "
    metrics_format: str = ".3e"


class _BatchesProcessedColumn(ProgressColumn):
    """Render the completed and total step counts in Lightning's Rich layout."""

    def __init__(self, style: str) -> None:
        """Initialize the counter column.

        Args:
            style: Rich style applied to the rendered counter.
        """
        self._style = style
        super().__init__()

    def render(self, task: Task) -> Text:
        """Render one task's completed and total step counts.

        Args:
            task: Rich progress task being displayed.

        Returns:
            Styled `completed/total` text.
        """
        total = "--" if task.total is None else f"{task.total:g}"
        return Text(f"{int(task.completed)}/{total}", style=self._style)


class _TimeColumn(ProgressColumn):
    """Render elapsed and remaining durations in Lightning's compact format."""

    max_refresh = 0.5

    def __init__(self, style: str) -> None:
        """Initialize the duration column.

        Args:
            style: Rich style applied to both durations.
        """
        self._style = style
        super().__init__()

    def render(self, task: Task) -> Text:
        """Render elapsed and estimated remaining durations.

        Args:
            task: Rich progress task being displayed.

        Returns:
            Styled `elapsed • remaining` text.
        """
        elapsed = task.finished_time if task.finished else task.elapsed
        elapsed_text = "-:--:--" if elapsed is None else str(timedelta(seconds=int(elapsed)))
        remaining = task.time_remaining
        remaining_text = "-:--:--" if remaining is None else str(timedelta(seconds=int(remaining)))
        return Text(f"{elapsed_text} • {remaining_text}", style=self._style)


class _ProcessingSpeedColumn(ProgressColumn):
    """Render training throughput in iterations per second."""

    def __init__(self, style: str) -> None:
        """Initialize the throughput column.

        Args:
            style: Rich style applied to the rendered throughput.
        """
        self._style = style
        super().__init__()

    def render(self, task: Task) -> Text:
        """Render one task's current iteration throughput.

        Args:
            task: Rich progress task being displayed.

        Returns:
            Styled iterations-per-second text.
        """
        speed = 0.0 if task.speed is None else task.speed
        return Text(f"{speed:>.2f}it/s", style=self._style)


class _MetricsColumn(ProgressColumn):
    """Render preformatted task metrics without touching accelerator values."""

    def __init__(self, style: str) -> None:
        """Initialize the metric column.

        Args:
            style: Rich style applied to the rendered metrics.
        """
        self._style = style
        super().__init__()

    def render(self, task: Task) -> Text:
        """Render the host metric text stored in the task fields.

        Args:
            task: Rich progress task being displayed.

        Returns:
            Styled metric text.
        """
        return Text(str(task.fields.get("metrics", "")), style=self._style)


class RichProgressBar(Callback):
    """Display fit and prediction progress with Rich.

    The batch counter advances without transferring device metrics to the host. Selected metric values are copied only
    on the first batch, every `refresh_rate` batches, and at fit completion. This keeps the progress display useful
    without introducing a device synchronization on every accelerator update.

    Attributes:
        total: Number of batches expected in one fit call.
        refresh_rate: Number of batches between metric-value refreshes.
        metric_names: Optional ordered override. `None` discovers `train/loss` and `train/weight` metrics automatically.
        description: Label displayed before the progress bar.
        predict_description: Label displayed during prediction.
        transient: Whether Rich removes the progress display after completion.
        rank_zero_only: Whether only JAX process zero renders the display.
        theme: Lightning-style colors and scalar formatting.
    """

    def __init__(
        self,
        total: int,
        *,
        refresh_rate: int = 10,
        metric_names: Sequence[str] | None = None,
        description: str = "Training",
        predict_description: str = "Predicting",
        transient: bool = False,
        rank_zero_only: bool = True,
        theme: RichProgressBarTheme | None = None,
        console: Console | None = None,
    ) -> None:
        """Initialize the Rich training progress display.

        Args:
            total: Number of batches expected in one fit call.
            refresh_rate: Positive interval between device-to-host metric transfers.
            metric_names: Optional ordered exact names to display. `None` selects combined and unweighted losses plus
                balancer weights published under the module's `train/loss` and `train/weight` metric namespaces.
            description: Non-empty label displayed before the progress bar.
            predict_description: Non-empty label displayed during prediction.
            transient: Whether Rich removes the progress display after completion.
            rank_zero_only: Whether only JAX process zero renders the display.
            theme: Optional Lightning-style colors and scalar formatting.
            console: Optional Rich console, primarily useful for embedding and tests.

        Raises:
            ValueError: If `total`, `refresh_rate`, either description, or a metric name is invalid.
        """
        if total < 1:
            raise ValueError("`total` must be positive.")
        if refresh_rate < 1:
            raise ValueError("`refresh_rate` must be positive.")
        if not description.strip():
            raise ValueError("`description` must not be empty.")
        if not predict_description.strip():
            raise ValueError("`predict_description` must not be empty.")
        names = None if metric_names is None else tuple(metric_names)
        if names is not None and any(not name.strip() for name in names):
            raise ValueError("`metric_names` must contain only non-empty names.")
        if names is not None and len(names) != len(set(names)):
            raise ValueError("`metric_names` must not contain duplicates.")
        resolved_theme = theme or RichProgressBarTheme()
        try:
            format(1.0, resolved_theme.metrics_format)
        except ValueError as error:
            raise ValueError("`theme.metrics_format` must be a valid scalar format specification.") from error
        self.total = total
        self.refresh_rate = refresh_rate
        self.metric_names = names
        self.description = description
        self.predict_description = predict_description
        self.transient = transient
        self.rank_zero_only = rank_zero_only
        self.theme = resolved_theme
        self._console = console
        self._progress: Progress | None = None
        self._task_id: TaskID | None = None
        self._completed = 0
        self._last_metrics = ""

    def setup(self) -> None:
        """Create fit-local Rich resources on the process responsible for display."""
        self._completed = 0
        self._last_metrics = ""
        self._task_id = None
        if self.rank_zero_only and jax.process_index() != 0:
            self._progress = None
            return
        self._progress = Progress(
            TextColumn("{task.description}", style=self.theme.description),
            BarColumn(
                complete_style=self.theme.progress_bar,
                finished_style=self.theme.progress_bar_finished,
                pulse_style=self.theme.progress_bar_pulse,
            ),
            _BatchesProcessedColumn(style=self.theme.batch_progress),
            _TimeColumn(style=self.theme.time),
            _ProcessingSpeedColumn(style=self.theme.processing_speed),
            _MetricsColumn(style=self.theme.metrics),
            console=self._console,
            transient=self.transient,
        )

    def on_fit_start(self, context: TrainerContext) -> None:
        """Start a new progress task before the first training batch.

        Args:
            context: Initial trainer context. Its global step may be nonzero after resume.
        """
        del context
        if self._progress is None:
            return
        self._progress.start()
        self._task_id = self._progress.add_task(self._step_description(0), total=self.total, metrics="")

    def on_train_batch_end(self, context: TrainerContext) -> bool:
        """Advance the bar and periodically refresh its selected metrics.

        Args:
            context: Post-update trainer context containing device metrics.

        Returns:
            Always `False`; progress reporting never requests early termination.
        """
        if self._progress is None or self._task_id is None:
            return False
        self._completed += 1
        should_refresh_metrics = self._completed == 1 or self._completed % self.refresh_rate == 0
        if should_refresh_metrics:
            self._last_metrics = self._format_metrics(context.metrics)
        self._progress.update(
            self._task_id,
            advance=1,
            description=self._step_description(self._completed),
            metrics=self._last_metrics,
        )
        return False

    def on_fit_end(self, context: TrainerContext) -> None:
        """Render terminal metrics and stop the progress display.

        Args:
            context: Final trainer context after a successful or callback-stopped fit.
        """
        if self._progress is None or self._task_id is None:
            return
        self._last_metrics = self._format_metrics(context.metrics)
        self._progress.update(self._task_id, metrics=self._last_metrics, refresh=True)
        self._stop()

    def on_predict_start(self, context: PredictionContext) -> None:
        """Start a progress task for one finite prediction source.

        Args:
            context: Initial prediction context containing the batch count when known.
        """
        if self._progress is None:
            return
        self._progress.start()
        self._task_id = self._progress.add_task(
            self._predict_description(0, context.total_batches),
            total=context.total_batches,
            metrics="",
        )

    def on_predict_batch_end(self, context: PredictionContext) -> None:
        """Advance the prediction bar after one valid output batch.

        Args:
            context: Completed prediction-batch context.
        """
        if self._progress is None or self._task_id is None:
            return
        self._completed += 1
        self._progress.update(
            self._task_id,
            advance=1,
            description=self._predict_description(self._completed, context.total_batches),
        )

    def on_predict_end(self, context: PredictionContext) -> None:
        """Render the completed prediction count and stop the progress display.

        Args:
            context: Final prediction context.
        """
        if self._progress is None or self._task_id is None:
            return
        self._progress.update(
            self._task_id,
            description=self._predict_description(self._completed, context.total_batches),
            refresh=True,
        )
        self._stop()

    def on_exception(self, exception: BaseException, context: CallbackContext) -> None:
        """Mark and stop the progress display before an exception is re-raised.

        Args:
            exception: Exception raised during training.
            context: Most recent valid callback context.
        """
        del exception, context
        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, description=f"{self.description} failed", refresh=True)
        self._stop()

    def teardown(self) -> None:
        """Stop Rich resources idempotently during trainer cleanup."""
        self._stop()

    def _format_metrics(self, metrics: Mapping[str, Any]) -> str:
        """Format configured scalar metrics after transferring only those values to the host.

        Args:
            metrics: Latest named device or host metrics.

        Returns:
            Space-separated `name=value` fields for metrics that are available.

        Raises:
            ValueError: If a selected available metric is not scalar.
        """
        fields: list[str] = []
        names = self.metric_names if self.metric_names is not None else self._automatic_metric_names(metrics)
        for name in names:
            if name not in metrics:
                continue
            array = np.asarray(jax.device_get(metrics[name]))
            if array.size != 1:
                raise ValueError(f"Progress metric `{name}` must be scalar, received shape {array.shape}.")
            scalar = float(array.reshape(()))
            fields.append(f"{name}: {scalar:{self.theme.metrics_format}}")
        return self.theme.metrics_text_delimiter.join(fields)

    def _automatic_metric_names(self, metrics: Mapping[str, Any]) -> tuple[str, ...]:
        """Select losses and balancer weights from a compiled-step metric mapping.

        Args:
            metrics: Latest named device or host metrics in their stable insertion order.

        Returns:
            Loss and balancer-weight metric names in stable insertion order.
        """
        return tuple(
            name for name in metrics if name == "train/loss" or name.startswith(("train/loss/", "train/weight/"))
        )

    def _step_description(self, completed: int) -> str:
        """Build the step-based progress description.

        Args:
            completed: Number of batches completed by this fit call.

        Returns:
            Description containing current and total fit steps.
        """
        return f"{self.description} {completed}/{self.total}"

    def _predict_description(self, completed: int, total: int | None) -> str:
        """Build the batch-based prediction description.

        Args:
            completed: Number of completed prediction batches.
            total: Total prediction batches, or `None` when the source length is unknown.

        Returns:
            Description containing the current and total prediction batch counts.
        """
        total_text = "--" if total is None else str(total)
        return f"{self.predict_description} {completed}/{total_text}"

    def _stop(self) -> None:
        """Stop and clear the current Rich progress resource if one exists."""
        if self._progress is not None:
            self._progress.stop()
        self._progress = None
        self._task_id = None


__all__ = ["RichProgressBar", "RichProgressBarTheme"]
