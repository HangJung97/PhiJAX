from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any


@dataclass
class ProgressTrainer:
    """Provide the Trainer attributes that progress-bar callbacks read.

    Attributes:
        max_steps: Global step at which fitting stops.
        progress_bar_metrics: Metrics a progress bar shows when no `metric_names` are selected.
        callback_metrics: Every routed metric, from which explicit `metric_names` are selected.
        logger: Logger stand-in whose `version` is shown as `v_num` when set.
    """

    max_steps: int
    progress_bar_metrics: dict[str, Any] = field(default_factory=dict)
    callback_metrics: dict[str, Any] = field(default_factory=dict)
    logger: SimpleNamespace = field(default_factory=lambda: SimpleNamespace(version=None))

    def route(self, metrics: dict[str, Any]) -> None:
        """Expose one step's metrics to both selection paths.

        Args:
            metrics: Metrics produced by the current step.
        """
        self.progress_bar_metrics = dict(metrics)
        self.callback_metrics = dict(metrics)
