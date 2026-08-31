from __future__ import annotations

import sys
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TextIO

import jax
import numpy as np
from tqdm.auto import tqdm

from phijax.callbacks.base import Callback, CallbackContext, PredictionContext, TrainerContext

if TYPE_CHECKING:
    from phijax.core import BasePhiModule
    from phijax.training.trainer import Trainer


class ProgressBar(Callback):
    """Define shared enablement and metric selection for progress displays."""

    def __init__(
        self,
        total: int,
        *,
        refresh_rate: int = 10,
        metric_names: Sequence[str] | None = None,
        description: str = "Training",
        predict_description: str = "Predicting",
        rank_zero_only: bool = True,
        metrics_format: str = ".3e",
    ) -> None:
        """Initialize common progress display policy.

        Args:
            total: Positive maximum optimizer updates in one fit call.
            refresh_rate: Positive number of batches between metric refreshes.
            metric_names: Optional ordered exact metric names overriding module-selected display metrics.
            description: Non-empty fit progress label.
            predict_description: Non-empty prediction progress label.
            rank_zero_only: Whether only JAX process zero writes output.
            metrics_format: Valid scalar format specification.

        Raises:
            ValueError: If a size, label, metric name, or scalar format is invalid.
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
        if names is not None and (any(not name.strip() for name in names) or len(names) != len(set(names))):
            raise ValueError("`metric_names` must contain unique non-empty names.")
        try:
            format(1.0, metrics_format)
        except ValueError as error:
            raise ValueError("`metrics_format` must be a valid scalar format specification.") from error
        self.total = total
        self.refresh_rate = refresh_rate
        self.metric_names = names
        self.description = description
        self.predict_description = predict_description
        self.rank_zero_only = rank_zero_only
        self.metrics_format = metrics_format
        self._enabled = True
        self._trainer: Trainer | None = None

    @property
    def is_enabled(self) -> bool:
        """Return whether rendering is enabled.

        Returns:
            Whether the callback may render output.
        """
        return self._enabled

    @property
    def is_disabled(self) -> bool:
        """Return whether rendering is disabled.

        Returns:
            Whether the callback suppresses output.
        """
        return not self.is_enabled

    def connect(self, trainer: Any) -> None:
        """Connect the progress callback to its Trainer.

        Args:
            trainer: Trainer exposing routed progress metrics and logger version.
        """
        self._trainer = trainer

    def disable(self) -> None:
        """Disable progress rendering."""
        self._enabled = False

    def enable(self) -> None:
        """Enable progress rendering."""
        self._enabled = True

    def get_metrics(self, trainer: Trainer, module: BasePhiModule | None) -> dict[str, Any]:
        """Combine standard Trainer fields with module-selected progress metrics.

        Args:
            trainer: Owning Trainer.
            module: Bound module participating in training, when available.

        Returns:
            Ordered host or device values selected for display.
        """
        del module
        metrics: dict[str, Any] = {}
        if trainer.logger.version is not None:
            metrics["v_num"] = str(trainer.logger.version)
        if self.metric_names is None:
            metrics.update(trainer.progress_bar_metrics)
        else:
            available = {**metrics, **trainer.callback_metrics}
            metrics = {name: available[name] for name in self.metric_names if name in available}
        return metrics

    def _formatted_metrics(self, context: TrainerContext) -> dict[str, str]:
        """Transfer and format selected progress metrics.

        Args:
            context: Complete post-update Trainer context.

        Returns:
            String values suitable for TQDM or Rich displays.

        Raises:
            ValueError: If a selected metric is not scalar.
        """
        if self._trainer is None:
            metrics = dict(context.metrics)
            if self.metric_names is None:
                metrics = {"train/loss": metrics["train/loss"]} if "train/loss" in metrics else {}
            else:
                metrics = {name: metrics[name] for name in self.metric_names if name in metrics}
        else:
            metrics = self.get_metrics(self._trainer, context.module)
        formatted: dict[str, str] = {}
        for name, value in metrics.items():
            if isinstance(value, str):
                formatted[name] = value
                continue
            array = np.asarray(jax.device_get(value))
            if array.size != 1:
                raise ValueError(f"Progress metric `{name}` must be scalar, received shape {array.shape}.")
            formatted[name] = format(float(array.reshape(())), self.metrics_format)
        return formatted


class TQDMProgressBar(ProgressBar):
    """Display compact Lightning-style fit and prediction progress with TQDM."""

    def __init__(self, total: int, *, stream: TextIO | None = None, **kwargs: Any) -> None:
        """Initialize a TQDM progress callback.

        Args:
            total: Positive maximum optimizer updates in one fit call.
            stream: Optional output stream, defaulting to the current `sys.stderr` during setup.
            **kwargs: Common progress options forwarded to :class:`ProgressBar`.
        """
        super().__init__(total, **kwargs)
        self._configured_stream = stream
        self._stream: TextIO | None = None
        self._bar: Any | None = None
        self._completed = 0

    def setup(self) -> None:
        """Reset task-local TQDM state and resolve rank-owned output."""
        owns_output = not self.rank_zero_only or jax.process_index() == 0
        self._stream = self._configured_stream or sys.stderr if owns_output else None
        self._bar = None
        self._completed = 0

    def on_fit_start(self, context: TrainerContext) -> None:
        """Create the fit progress display.

        Args:
            context: Initial Trainer context.
        """
        if self.is_disabled or self._stream is None or (self.rank_zero_only and not context.is_global_zero):
            return
        self._bar = tqdm(
            total=self.total,
            desc=self.description,
            file=self._stream,
            leave=True,
            dynamic_ncols=True,
            miniters=self.refresh_rate,
        )

    def on_train_metrics(self, context: TrainerContext) -> None:
        """Advance training progress and periodically refresh selected metrics.

        Args:
            context: Complete post-update metric context.
        """
        if self._bar is None:
            return
        self._completed += 1
        self._bar.update(1)
        if self._completed == 1 or self._completed % self.refresh_rate == 0 or self._completed == self.total:
            self._bar.set_postfix(self._formatted_metrics(context), refresh=True)

    def on_fit_end(self, context: TrainerContext) -> None:
        """Render terminal fit metrics and close the bar.

        Args:
            context: Terminal fit context.
        """
        if self._bar is not None:
            self._bar.set_postfix(self._formatted_metrics(context), refresh=False)
        self._close()

    def on_predict_start(self, context: PredictionContext) -> None:
        """Create the prediction progress display.

        Args:
            context: Initial prediction context.
        """
        if self.is_disabled or self._stream is None or (self.rank_zero_only and not context.is_global_zero):
            return
        self._bar = tqdm(
            total=context.total_batches,
            desc=self.predict_description,
            file=self._stream,
            leave=True,
            dynamic_ncols=True,
            miniters=self.refresh_rate,
        )

    def on_predict_batch_end(self, context: PredictionContext) -> None:
        """Advance prediction progress by one batch.

        Args:
            context: Completed prediction context.
        """
        del context
        if self._bar is not None:
            self._bar.update(1)

    def on_predict_end(self, context: PredictionContext) -> None:
        """Close the completed prediction display.

        Args:
            context: Terminal prediction context.
        """
        del context
        self._close()

    def on_exception(self, exception: BaseException, context: CallbackContext) -> None:
        """Mark a failed task and close its display.

        Args:
            exception: Exception terminating the task.
            context: Most recent callback context.
        """
        del exception
        if self._bar is not None:
            label = self.predict_description if isinstance(context, PredictionContext) else self.description
            self._bar.set_description_str(f"{label} failed", refresh=True)
        self._close()

    def teardown(self) -> None:
        """Close TQDM resources idempotently."""
        self._close()
        self._stream = None

    def _close(self) -> None:
        """Close and clear the current TQDM resource."""
        if self._bar is not None:
            self._bar.close()
        self._bar = None


__all__ = ["ProgressBar", "TQDMProgressBar"]
