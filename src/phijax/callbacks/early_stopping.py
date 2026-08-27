from phijax.callbacks.base import Callback, TrainerContext


class EarlyStopping(Callback):
    """Stop training when a monitored scalar metric ceases to improve.

    Attributes:
        monitor: Metric name inspected after each training batch.
        patience: Number of non-improving observations tolerated.
        mode: Whether lower (`min`) or higher (`max`) values improve the metric.
        min_delta: Required absolute change to count as an improvement.
    """

    def __init__(
        self,
        monitor: str = "train/loss",
        patience: int = 100,
        mode: str = "min",
        min_delta: float = 0.0,
    ) -> None:
        """Initialize metric-based early stopping.

        Args:
            monitor: Metric name inspected after each training batch.
            patience: Number of non-improving observations tolerated.
            mode: Either `min` or `max`.
            min_delta: Required absolute improvement over the current best.

        Raises:
            ValueError: If `patience`, `mode`, or `min_delta` is invalid.
        """
        if patience < 0:
            raise ValueError("`patience` must be non-negative.")
        if mode not in {"min", "max"}:
            raise ValueError("`mode` must be either `min` or `max`.")
        if min_delta < 0.0:
            raise ValueError("`min_delta` must be non-negative.")
        self.monitor = monitor
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.best: float | None = None
        self.bad_checks = 0

    def on_train_batch_end(self, context: TrainerContext) -> bool:
        """Update the monitored best value and request stopping after patience expires.

        Args:
            context: Trainer context containing the latest metric mapping.

        Returns:
            Whether the configured patience has expired.

        Raises:
            KeyError: If the monitored metric is absent.
            ValueError: If the monitored metric is not scalar.
        """
        if self.monitor not in context.metrics:
            raise KeyError(f"Monitored metric `{self.monitor}` was not produced by the training step.")
        value = context.metrics[self.monitor]
        if getattr(value, "size", 1) != 1:
            raise ValueError(f"Monitored metric `{self.monitor}` must be scalar.")
        scalar = float(value)
        improved = self.best is None
        if self.best is not None:
            if self.mode == "min":
                improved = scalar < self.best - self.min_delta
            else:
                improved = scalar > self.best + self.min_delta
        if improved:
            self.best = scalar
            self.bad_checks = 0
        else:
            self.bad_checks += 1
        return self.bad_checks > self.patience


__all__ = ["EarlyStopping"]
