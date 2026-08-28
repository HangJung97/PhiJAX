from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import jax

from phijax.types import ModelApply, NamedBatches


@runtime_checkable
class ObjectiveTerm(Protocol):
    """Define one independently configurable group of named objective losses."""

    @property
    def loss_names(self) -> Sequence[str]:
        """Return names produced by this term.

        Returns:
            Ordered unique loss names.
        """
        ...

    def losses(
        self,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> Mapping[str, jax.Array]:
        """Evaluate this term's named scalar losses.

        Args:
            model_apply: Pure explicit-state model application callable.
            model_state: Differentiable model parameter PyTree.
            batches: Fixed-structure named input batches.

        Returns:
            Scalar losses keyed by :attr:`loss_names`.
        """
        ...

    def residual_stream(
        self,
        name: str,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> jax.Array:
        """Evaluate one raw stream for derivative-based balancing.

        Args:
            name: One entry from :attr:`loss_names`.
            model_apply: Pure explicit-state model application callable.
            model_state: Differentiable model parameter PyTree.
            batches: Fixed-structure named residual batches.

        Returns:
            Raw output or equation residual array.
        """
        ...


@runtime_checkable
class Objective(Protocol):
    """Define the scalar-loss contract consumed by compiled training steps."""

    @property
    def loss_names(self) -> Sequence[str]:
        """Return the stable names used to align losses and balancer weights.

        Returns:
            Ordered unique loss names.
        """
        ...

    def losses(
        self,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> Mapping[str, jax.Array]:
        """Evaluate named scalar losses.

        Args:
            model_apply: Pure explicit-state model application callable.
            model_state: Differentiable model parameter PyTree.
            batches: Fixed-structure named input batches.

        Returns:
            Scalar loss arrays keyed by :attr:`loss_names`.
        """
        ...


@runtime_checkable
class ResidualObjective(Objective, Protocol):
    """Extend an objective with one-at-a-time residual streams for derivative-based balancing."""

    def residual_stream(
        self,
        name: str,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> jax.Array:
        """Evaluate one raw residual stream.

        Args:
            name: One entry from :attr:`loss_names`.
            model_apply: Pure explicit-state model application callable.
            model_state: Differentiable model parameter PyTree.
            batches: Fixed-structure named residual batches.

        Returns:
            Residual array whose parameter Jacobian defines the stream's empirical NTK diagnostics.
        """
        ...


__all__ = ["Objective", "ObjectiveTerm", "ResidualObjective"]
