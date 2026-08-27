from collections.abc import Callable, Mapping
from typing import Any, Literal, Protocol

import jax

type ArrayMapping = Mapping[str, jax.Array]
type NamedBatches = Mapping[str, ArrayMapping]
type ModelApply = Callable[[Any, jax.Array], jax.Array]
type ResidualStream = Literal["residual", "output"]
type ResidualGroup = tuple[jax.Array, ...]
type ResidualGroups = tuple[ResidualGroup, ...]


class JaxDevice(Protocol):
    """Describe the stable JAX device metadata used by PhiJAX orchestration.

    JAX exposes its concrete device class dynamically, which some static type checkers reject in annotations. This
    protocol keeps PhiJAX independent of private :mod:`jaxlib` implementation modules while accepting real JAX devices.

    Attributes:
        platform: Generic device platform such as `cpu`, `gpu`, or `tpu`.
        process_index: JAX process that owns the device.
        id: Backend-local device identifier.
    """

    platform: str
    process_index: int
    id: int


class ModelSummaryFunction(Protocol):
    """Define a model-state summary callable independent of a concrete neural-network library."""

    def __call__(
        self,
        model_state: Any,
        *,
        max_depth: int = -1,
        console_width: int = 120,
        compute_flops: bool = False,
        compute_vjp_flops: bool = False,
    ) -> str:
        """Render a model summary for explicit model state.

        Args:
            model_state: Explicit model parameter and variable state.
            max_depth: Maximum displayed module depth, or `-1` for every level.
            console_width: Positive Rich console width in characters.
            compute_flops: Whether to estimate forward-pass floating-point operations.
            compute_vjp_flops: Whether to estimate reverse-pass floating-point operations.

        Returns:
            Rendered model-summary table.
        """
        ...


class ResidualFunction(Protocol):
    """Define one configured equation callable that produces grouped residual streams."""

    def __call__(
        self,
        model_apply: ModelApply,
        model_state: Any,
        batch: ArrayMapping,
        *,
        stream: ResidualStream = "residual",
    ) -> ResidualGroups:
        """Evaluate residual groups or an explicitly supported alternative stream.

        Args:
            model_apply: Pure explicit-state model application callable.
            model_state: Differentiable model parameter PyTree.
            batch: Fixed-structure arrays required by the configured equation.
            stream: Stream representation requested by the objective.

        Returns:
            Residual groups aligned with the owning objective term's loss names.
        """
        ...


__all__ = [
    "ArrayMapping",
    "JaxDevice",
    "ModelApply",
    "ModelSummaryFunction",
    "NamedBatches",
    "ResidualFunction",
    "ResidualGroup",
    "ResidualGroups",
    "ResidualStream",
]
