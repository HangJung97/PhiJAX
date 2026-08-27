from collections.abc import Callable
from dataclasses import dataclass

from phijax.balancers import BalancerUpdatePlan
from phijax.training.steps import TrainStep


@dataclass(frozen=True, slots=True)
class BalancerUpdateSchedule:
    """Describe when the Trainer applies one adaptive-balancer update plan.

    Attributes:
        plan: Functional adaptive update and its diagnostic batch-size policy.
        every_n_steps: Positive optimizer-step interval between updates.
        skip_first_step: Whether to avoid updating before the first optimizer step.
    """

    plan: BalancerUpdatePlan
    every_n_steps: int
    skip_first_step: bool = True

    def __post_init__(self) -> None:
        """Validate host-side adaptive update scheduling.

        Raises:
            TypeError: If scheduling values have invalid types.
            ValueError: If `every_n_steps` is not positive.
        """
        if isinstance(self.every_n_steps, bool) or not isinstance(self.every_n_steps, int):
            raise TypeError("`every_n_steps` must be an integer.")
        if self.every_n_steps < 1:
            raise ValueError("`every_n_steps` must be positive.")
        if not isinstance(self.skip_first_step, bool):
            raise TypeError("`skip_first_step` must be Boolean.")


@dataclass(frozen=True, slots=True)
class TrainingPlan:
    """Bind a compiled optimizer step to its data and adaptive-update requirements.

    Attributes:
        train_step: Compiled state-and-batch optimizer update.
        batch_keys: Stable named batches requested from a DataModule.
        balancer_update: Optional host-scheduled adaptive-balancer update.
    """

    train_step: TrainStep
    batch_keys: tuple[str, ...] = ()
    balancer_update: BalancerUpdateSchedule | None = None

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


__all__ = ["BalancerUpdateSchedule", "TrainingPlan"]
