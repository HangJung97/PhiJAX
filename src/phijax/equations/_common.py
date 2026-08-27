from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from phijax.types import ModelApply, ResidualStream


def evaluate_selected_outputs(
    model_apply: ModelApply,
    model_state: Any,
    inputs: jax.Array,
    indices: Sequence[int],
) -> jax.Array:
    """Evaluate a model batch and select static trailing output components.

    Args:
        model_apply: Pure explicit-state model application callable.
        model_state: Differentiable model parameter PyTree.
        inputs: Batched model inputs.
        indices: Non-empty unique model-output component indices.

    Returns:
        Batched selected output components.

    Raises:
        ValueError: If `indices` contains invalid components.
    """
    outputs = jax.vmap(model_apply, in_axes=(None, 0))(model_state, inputs)
    return select_components(outputs, indices, option="output_indices")


def select_components(values: jax.Array, indices: Sequence[int], *, option: str) -> jax.Array:
    """Select validated static components from an array's trailing axis.

    Args:
        values: Array containing a trailing component axis.
        indices: Non-empty unique component indices.
        option: Configuration option name used in validation errors.

    Returns:
        Array retaining selected components in configured order.

    Raises:
        ValueError: If `indices` contains negative or duplicated components.
    """
    resolved_indices = validate_component_indices(indices, option=option)
    return jnp.take(values, jnp.asarray(resolved_indices), axis=-1)


def validate_component_indices(indices: Sequence[int], *, option: str) -> tuple[int, ...]:
    """Validate static component indices.

    Args:
        indices: Candidate component indices.
        option: Configuration option name used in errors.

    Returns:
        Immutable validated component indices.

    Raises:
        ValueError: If indices are empty, negative, or duplicated.
    """
    resolved = tuple(indices)
    if not resolved or any(index < 0 for index in resolved) or len(set(resolved)) != len(resolved):
        raise ValueError(f"`{option}` must contain unique nonnegative component indices.")
    return resolved


def validate_stream(stream: ResidualStream, *, supports_output: bool) -> None:
    """Validate a requested residual representation.

    Args:
        stream: Requested equation representation.
        supports_output: Whether the equation exposes selected raw model outputs.

    Raises:
        ValueError: If `stream` is unknown or requests an unsupported output representation.
    """
    if stream not in ("residual", "output"):
        raise ValueError("`stream` must be either 'residual' or 'output'.")
    if stream == "output" and not supports_output:
        raise ValueError("This equation does not expose a raw model-output stream.")


__all__ = ["evaluate_selected_outputs", "select_components", "validate_component_indices", "validate_stream"]
