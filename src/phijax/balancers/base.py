from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

import jax
from flax import struct

from phijax.types import NamedBatches

type BalancerUpdate = Callable[[Any, NamedBatches, Any], Any]


@runtime_checkable
class LossBalancer(Protocol):
    """Define the functional contract for combining named scalar losses."""

    loss_names: tuple[str, ...]

    def initialize(self) -> Any:
        """Create the JAX-compatible initial balancer state.

        Returns:
            Initial balancer-state PyTree.
        """
        ...

    def combine(self, losses: Mapping[str, jax.Array], state: Any) -> jax.Array:
        """Combine named unweighted losses into one scalar objective.

        Args:
            losses: Scalar losses keyed by every configured loss name.
            state: Current balancer-state PyTree.

        Returns:
            Weighted scalar loss.
        """
        ...

    def diagnostics(self, state: Any) -> Mapping[str, jax.Array]:
        """Expose scalar diagnostics without interpreting state externally.

        Args:
            state: Current balancer-state PyTree.

        Returns:
            Scalar diagnostics with stable names.
        """
        ...


@dataclass(frozen=True, slots=True)
class BalancerUpdatePlan:
    """Describe one host-scheduled adaptive-balancer update.

    Attributes:
        update: Compiled functional update accepting model state, named batches, and balancer state.
        every_n_steps: Positive optimizer-step interval between updates.
        update_start_step: Nonnegative absolute optimizer step anchoring the update cadence.
        batch_sizes: Optional positive diagnostic batch size for each objective batch key. `None` reuses the current
            training batches instead of sampling a fixed diagnostic batch.
    """

    update: BalancerUpdate
    every_n_steps: int
    update_start_step: int
    batch_sizes: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        """Validate host scheduling and fixed diagnostic-batch sizes.

        Raises:
            TypeError: If scheduling values or diagnostic batch names have invalid types.
            ValueError: If an interval or diagnostic batch size is zero or negative.
        """
        if not callable(self.update):
            raise TypeError("`update` must be callable.")
        if isinstance(self.every_n_steps, bool) or not isinstance(self.every_n_steps, int):
            raise TypeError("`every_n_steps` must be an integer.")
        if self.every_n_steps < 1:
            raise ValueError("`every_n_steps` must be positive.")
        if isinstance(self.update_start_step, bool) or not isinstance(self.update_start_step, int):
            raise TypeError("`update_start_step` must be an integer.")
        if self.update_start_step < 0:
            raise ValueError("`update_start_step` must be nonnegative.")
        if self.batch_sizes is None:
            return
        if any(not isinstance(name, str) or not name.strip() for name in self.batch_sizes):
            raise TypeError("Adaptive balancer diagnostic batch names must be non-empty strings.")
        invalid = {
            name: size
            for name, size in self.batch_sizes.items()
            if isinstance(size, bool) or not isinstance(size, int) or size < 1
        }
        if invalid:
            raise ValueError(f"Adaptive balancer diagnostic batch sizes must be positive integers, got {invalid}.")
        object.__setattr__(self, "batch_sizes", MappingProxyType(dict(self.batch_sizes)))


@runtime_checkable
class AdaptiveBalancer(Protocol):
    """Define the assembly contract for a periodically updated loss balancer."""

    def build_update_plan(
        self,
        module: Any,
        batch_keys: Sequence[str],
    ) -> BalancerUpdatePlan:
        """Build a reusable update and declare its diagnostic data policy.

        Args:
            module: Application module exposing the numerical streams required by the balancer.
            batch_keys: Stable objective batch keys available from the training source.

        Returns:
            Functional update, scheduling, and diagnostic data plan consumed by training assembly.
        """
        ...


@struct.dataclass
class BalancerState:
    """Store JIT-compatible loss weights and the most recent NTK diagnostics.

    Attributes:
        weights: Vector aligned with the balancer's fixed loss-name ordering.
        traces: Most recently computed NTK statistic for each loss, or zeros before the first update. Exact-NTK
            balancing stores the mean pointwise diagonal empirical NTK in this field.
    """

    weights: jax.Array
    traces: jax.Array


__all__ = ["AdaptiveBalancer", "BalancerState", "BalancerUpdate", "BalancerUpdatePlan", "LossBalancer"]
