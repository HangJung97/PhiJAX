import math
from collections.abc import Callable, Iterable, Mapping
from typing import Any, Literal, NamedTuple, Protocol

import jax

type ArrayMapping = Mapping[str, jax.Array]
type NamedBatches = Mapping[str, ArrayMapping]
type ModelApply = Callable[[Any, jax.Array], jax.Array]
type ResidualStream = Literal["residual", "output"]
type ResidualGroup = tuple[jax.Array, ...]
type ResidualGroups = tuple[ResidualGroup, ...]
# Hashable normalized form stored in models, modules, and training state: sorted `(axis, period)` pairs.
type ResolvedInputPeriods = tuple[tuple[int, float], ...]
# Physical period of each periodic input axis, keyed by axis index, or already normalized pairs.
type InputPeriods = Mapping[int, float] | ResolvedInputPeriods


class InputStatistics(NamedTuple):
    """Hold per-coordinate model-input normalization statistics.

    The Trainer stores the statistics in :class:`phijax.training.TrainState` so checkpoints restore the normalization
    a model was trained with. A NamedTuple keeps tuple unpacking, `mean, std = statistics`, working.

    Attributes:
        mean: Per-coordinate mean with shape `[input_dim]`.
        std: Per-coordinate standard deviation with shape `[input_dim]`.
    """

    mean: Any
    std: Any


def as_input_periods(
    periods: Mapping[Any, Any] | Iterable[tuple[Any, Any]] | None,
) -> ResolvedInputPeriods | None:
    """Normalize per-axis physical periods to sorted `(axis, period)` pairs of Python numbers.

    Axis keys are converted with `int()`, so string keys from configuration overrides are accepted.

    Args:
        periods: Mapping or pairs from input axis to physical period, or `None`.

    Returns:
        Sorted hashable pairs, or `None` when no periods are supplied.

    Raises:
        ValueError: If an axis is negative or repeated, or a period is not finite and positive.
    """
    if periods is None:
        return None
    resolved: dict[int, float] = {}
    pairs: list[tuple[Any, Any]] = list(periods.items() if isinstance(periods, Mapping) else periods)
    for axis, period in pairs:
        index, value = int(axis), float(period)
        if index < 0 or index in resolved:
            raise ValueError(f"Periodic axes must be unique nonnegative integers; got `{axis}`.")
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"Periods must be finite and positive; axis {index} has period {period}.")
        resolved[index] = value
    return tuple(sorted(resolved.items())) or None


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
    "InputPeriods",
    "InputStatistics",
    "JaxDevice",
    "ModelApply",
    "ModelSummaryFunction",
    "NamedBatches",
    "ResidualFunction",
    "ResidualGroup",
    "ResidualGroups",
    "ResidualStream",
    "ResolvedInputPeriods",
    "as_input_periods",
]
