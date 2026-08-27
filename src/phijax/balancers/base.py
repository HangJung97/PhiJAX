from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
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
        batch_sizes: Optional positive diagnostic batch size for each objective batch key. `None` reuses the current
            training batches instead of sampling a fixed diagnostic batch.
    """

    update: BalancerUpdate
    batch_sizes: Mapping[str, int] | None = None

    def __post_init__(self) -> None:
        """Validate fixed diagnostic-batch sizes.

        Raises:
            ValueError: If a configured diagnostic batch size is Boolean, zero, or negative.
        """
        if self.batch_sizes is None:
            return
        invalid = {
            name: size
            for name, size in self.batch_sizes.items()
            if isinstance(size, bool) or not isinstance(size, int) or size < 1
        }
        if invalid:
            raise ValueError(f"Adaptive balancer diagnostic batch sizes must be positive integers, got {invalid}.")


@runtime_checkable
class AdaptiveBalancer(Protocol):
    """Define the assembly contract for a periodically updated loss balancer."""

    def build_update_plan(
        self,
        module: Any,
        batch_keys: Sequence[str],
        options: Mapping[str, Any],
    ) -> BalancerUpdatePlan:
        """Build a reusable update and declare its diagnostic data policy.

        Args:
            module: Application module exposing the numerical streams required by the balancer.
            batch_keys: Stable objective batch keys available from the training source.
            options: Balancer-specific resolved options from `model.balancer.update` after removing scheduling fields.

        Returns:
            Functional update plan consumed generically by training assembly.
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
