from dataclasses import dataclass

from phijax.core import BasePhiModule
from phijax.training.state import TrainState


@dataclass(frozen=True, slots=True)
class FitResult:
    """Summarize the terminal state of one Trainer fit call.

    Attributes:
        module: Bound module used for training and subsequent prediction.
        state: Final functional training state.
        metrics: Final host scalar metrics.
        stopped_early: Whether a callback requested termination.
        iterations: Number of batches processed by this fit call.
        interrupted: Whether an operating-system signal or :class:`KeyboardInterrupt` stopped training.
    """

    module: BasePhiModule
    state: TrainState
    metrics: dict[str, float]
    stopped_early: bool
    iterations: int
    interrupted: bool = False


__all__ = ["FitResult"]
