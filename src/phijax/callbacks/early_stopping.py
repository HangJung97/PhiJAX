import logging
import math
import warnings
from collections.abc import Mapping
from typing import Any

from phijax.callbacks.base import Callback, MonitorMode, TrainerContext

log = logging.getLogger(__name__)


class EarlyStopping(Callback):
    """Stop training when a monitored scalar metric stops improving, following Lightning's `EarlyStopping`.

    The callback checks the metric in :meth:`on_train_metrics`, after module-logged and callback metrics are merged,
    and requests stopping through :attr:`phijax.training.Trainer.should_stop`.

    Attributes:
        wait_count: Consecutive checks without improvement.
        best_score: Best monitored value so far, or `None` before the first check.
        stopped_step: Global step at which this callback requested stopping, or `0` if it has not.
    """

    def __init__(
        self,
        monitor: str = "train/loss",
        patience: int = 100,
        mode: MonitorMode = "min",
        min_delta: float = 0.0,
        check_finite: bool = True,
        check_every_n_steps: int = 1,
    ) -> None:
        """Initialize metric-based early stopping.

        Args:
            monitor: Scalar metric checked after each selected training step.
            patience: Number of checks without improvement after which training stops. As in Lightning, training
                stops once `wait_count >= patience`.
            mode: `min` when smaller values are better, or `max` when larger values are better.
            min_delta: Smallest absolute change over `best_score` that counts as an improvement.
            check_finite: Whether to stop immediately when the monitored value is NaN or infinite.
            check_every_n_steps: Positive interval of global steps between checks. Each check transfers the metric to
                the host, so a larger interval avoids a device synchronization on every step.

        Raises:
            ValueError: If `patience`, `mode`, `min_delta`, or `check_every_n_steps` is invalid.
        """
        if patience < 0:
            raise ValueError("`patience` must be non-negative.")
        if mode not in {"min", "max"}:
            raise ValueError("`mode` must be either `min` or `max`.")
        if min_delta < 0.0:
            raise ValueError("`min_delta` must be non-negative.")
        if isinstance(check_every_n_steps, bool) or not isinstance(check_every_n_steps, int) or check_every_n_steps < 1:
            raise ValueError("`check_every_n_steps` must be a positive integer.")
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.check_finite = check_finite
        self.check_every_n_steps = check_every_n_steps
        self.wait_count = 0
        self.best_score: float | None = None
        self.stopped_step = 0
        self._trainer: Any | None = None

    def connect(self, trainer: Any) -> None:
        """Keep the Trainer whose `should_stop` flag this callback sets.

        Args:
            trainer: Host Trainer coordinating callback hooks.
        """
        self._trainer = trainer

    def on_train_metrics(self, context: TrainerContext) -> None:
        """Check the monitored metric and request stopping when patience runs out.

        Args:
            context: Post-update context containing module-logged and callback metrics.

        Raises:
            KeyError: If the monitored metric is absent.
            ValueError: If the monitored metric is not scalar.
        """
        if context.step % self.check_every_n_steps != 0:
            return
        if self.monitor not in context.metrics:
            raise KeyError(f"Monitored metric `{self.monitor}` was not produced by the training step.")
        value = context.metrics[self.monitor]
        if getattr(value, "size", 1) != 1:
            raise ValueError(f"Monitored metric `{self.monitor}` must be scalar.")
        current = float(value)
        if self.check_finite and not math.isfinite(current):
            reason = f"Monitored metric `{self.monitor}` = {current} is not finite."
        elif self._improved(current):
            self.best_score = current
            self.wait_count = 0
            return
        else:
            self.wait_count += 1
            if self.wait_count < self.patience:
                return
            reason = (
                f"Monitored metric `{self.monitor}` did not improve in the last {self.wait_count} checks. "
                f"Best score: {self.best_score}."
            )
        self.stopped_step = context.step
        if self._trainer is not None:
            self._trainer.should_stop = True
        if context.is_global_zero:
            log.info(f"EarlyStopping stopped training at step {context.step}. {reason}")

    def _improved(self, current: float) -> bool:
        """Return whether `current` improves on `best_score` by more than `min_delta`.

        Args:
            current: Finite or non-finite monitored value.

        Returns:
            Whether the value counts as an improvement.
        """
        if self.best_score is None:
            return math.isfinite(current)
        if self.mode == "min":
            return current < self.best_score - self.min_delta
        return current > self.best_score + self.min_delta

    def state_dict(self) -> Mapping[str, Any]:
        """Return the progress needed to resume early stopping exactly, with the settings it was measured under.

        Returns:
            JSON-compatible progress (`wait_count`, `best_score`, `stopped_step`) and settings (`monitor`, `mode`,
            `patience`).
        """
        return {
            "wait_count": self.wait_count,
            "best_score": self.best_score,
            "stopped_step": self.stopped_step,
            "monitor": self.monitor,
            "mode": self.mode,
            "patience": self.patience,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore early-stopping progress from a checkpoint.

        Progress recorded for a different `monitor` or `mode` is not comparable, so it is skipped with a warning. The
        configured `patience` always wins; a different stored value only produces a warning. Checkpoints without
        stored settings restore their progress.

        Args:
            state: Mapping produced by :meth:`state_dict`.

        Raises:
            KeyError: If a required progress entry is missing.
        """
        stored = (state.get("monitor", self.monitor), state.get("mode", self.mode))
        if stored != (self.monitor, self.mode):
            warnings.warn(
                f"EarlyStopping: skipping stored progress for monitor={stored[0]!r}, mode={stored[1]!r}; "
                f"this callback monitors {self.monitor!r} in {self.mode!r} mode and starts fresh.",
                UserWarning,
                stacklevel=2,
            )
            return
        stored_patience = state.get("patience", self.patience)
        if stored_patience != self.patience:
            warnings.warn(
                f"EarlyStopping: using the configured patience={self.patience}; "
                f"the checkpoint was trained with patience={stored_patience}.",
                UserWarning,
                stacklevel=2,
            )
        self.wait_count = int(state["wait_count"])
        self.best_score = None if state["best_score"] is None else float(state["best_score"])
        self.stopped_step = int(state["stopped_step"])


__all__ = ["EarlyStopping"]
