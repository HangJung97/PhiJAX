from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp

from phijax.equations._common import (
    evaluate_selected_outputs,
    select_components,
    validate_component_indices,
    validate_stream,
)
from phijax.equations.metadata import residual_equation
from phijax.types import ArrayMapping, ModelApply, ResidualGroups, ResidualStream


def base_boundary_residual(output: jax.Array, target: jax.Array) -> jax.Array:
    """Return a direct supervised boundary residual.

    Args:
        output: Predicted boundary values with shape `[..., components]`.
        target: Reference boundary values broadcast-compatible with `output` and the same final width.

    Returns:
        Direct `output - target` residual with the broadcast input shape.

    Raises:
        ValueError: If `output` and `target` final widths do not match.
    """
    if output.shape[-1] != target.shape[-1]:
        raise ValueError("Boundary output and target widths must match.")
    return output - target


def no_slip_residual(velocity: jax.Array, target: jax.Array) -> jax.Array:
    """Constrain every velocity component to a reference wall velocity.

    Args:
        velocity: Predicted velocity components with shape `[..., dimensions]`.
        target: Reference wall velocity broadcast-compatible with `velocity`.

    Returns:
        Component-wise no-slip residual with the broadcast input shape.

    Raises:
        ValueError: If the final velocity and target widths do not match.
    """
    return base_boundary_residual(velocity, target)


def free_slip_residual(velocity: jax.Array, target: jax.Array, normals: jax.Array) -> jax.Array:
    """Project velocity mismatch onto the wall-normal direction.

    Args:
        velocity: Predicted velocity components with shape `[..., dimensions]`.
        target: Reference wall velocity broadcast-compatible with `velocity`.
        normals: Wall-normal components with the same final width as `velocity`.

    Returns:
        Scalar normal residual per sample with shape `[..., 1]`.

    Raises:
        ValueError: If the final velocity, target, and normal dimensions do not match.
    """
    velocity_residual = no_slip_residual(velocity, target)
    if velocity_residual.shape[-1] != normals.shape[-1]:
        raise ValueError("Velocity and normal widths must match.")
    return jnp.sum(velocity_residual * normals, axis=-1, keepdims=True)


@residual_equation(names=("free_slip",), default_ntk_stream="output")
def free_slip_boundary(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    output_indices: Sequence[int] = (0, 1),
    target_indices: Sequence[int] = (0, 1),
    stream: ResidualStream = "residual",
) -> ResidualGroups:
    """Evaluate grouped free-slip boundary residuals or constrained outputs.

    Args:
        model_apply: Pure explicit-state model application callable.
        model_state: Differentiable model parameter PyTree.
        batch: Arrays containing `inputs`, `targets`, and `normals`.
        output_indices: Model-output velocity components constrained at the boundary.
        target_indices: Target velocity components aligned with `output_indices`.
        stream: `"residual"` for the normal-velocity error or `"output"` for selected model outputs.

    Returns:
        One residual group for scalar loss reduction, or one selected-output group for output-based NTK diagnostics.

    Raises:
        ValueError: If the stream or component selection is invalid.
    """
    validate_stream(stream, supports_output=True)
    resolved_outputs = validate_component_indices(output_indices, option="output_indices")
    resolved_targets = validate_component_indices(target_indices, option="target_indices")
    if len(resolved_outputs) != len(resolved_targets):
        raise ValueError("`output_indices` and `target_indices` must have equal lengths.")
    output = evaluate_selected_outputs(model_apply, model_state, batch["inputs"], resolved_outputs)
    if stream == "output":
        return ((output,),)
    target = select_components(batch["targets"], resolved_targets, option="target_indices")
    return ((free_slip_residual(output, target, batch["normals"]),),)


__all__ = ["base_boundary_residual", "free_slip_boundary", "free_slip_residual", "no_slip_residual"]
