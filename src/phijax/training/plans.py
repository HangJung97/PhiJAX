from collections.abc import Callable
from dataclasses import dataclass

from phijax.balancers import BalancerUpdatePlan
from phijax.training.steps import TrainStep


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """Bind a compiled optimizer step to its data and adaptive-update requirements.

    Attributes:
        train_step: Compiled state-and-batch optimizer update that increments `TrainState.step` exactly once.
        batch_keys: Stable named batches requested from a DataModule.
        balancer_update: Optional host-scheduled adaptive-balancer update.
    """

    train_step: TrainStep
    batch_keys: tuple[str, ...] = ()
    balancer_update: BalancerUpdatePlan | None = None

    def __post_init__(self) -> None:
        """Validate the immutable plan structure.

        Raises:
            TypeError: If `train_step` is not callable.
            ValueError: If batch keys are empty or duplicated.
        """
        if not isinstance(self.train_step, Callable):
            raise TypeError("`train_step` must be callable.")
        if any(not name or not name.strip() for name in self.batch_keys):
            raise ValueError("Training-plan batch keys must be non-empty strings.")
        if len(set(self.batch_keys)) != len(self.batch_keys):
            raise ValueError("Training-plan batch keys must be unique.")


__all__ = ["TrainingPlan"]
