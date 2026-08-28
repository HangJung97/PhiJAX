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


def base_data_fidelity_residual(
    output: jax.Array,
    target: jax.Array,
    *,
    weight: jax.Array | None = None,
    projection: jax.Array | None = None,
    target_negation: bool = False,
) -> jax.Array:
    """Return an unwrapped supervised residual.

    Args:
        output: Predicted scalar or vector values.
        target: Observed values broadcast-compatible with the resolved output.
        weight: Optional multiplicative residual weights applied after projection.
        projection: Optional direction with the same shape as `output`. When supplied, the vector output is projected
            to a trailing singleton dimension before comparison with `target`.
        target_negation: Whether to negate targets before subtraction.

    Returns:
        Weighted `output - signed_target` residual with the broadcast input shape.

    Raises:
        ValueError: If `output` and `projection` shapes do not match.
    """
    resolved_output = output
    if projection is not None:
        if output.shape != projection.shape:
            raise ValueError(
                "Projected data fidelity requires output and projection arrays to have matching shapes: "
                f"got output={output.shape} and projection={projection.shape}."
            )
        resolved_output = jnp.sum(output * projection, axis=-1, keepdims=True)
    signed_target = -target if target_negation else target
    residual = resolved_output - signed_target
    return residual if weight is None else weight * residual


@residual_equation(names=("data",))
def base_data_fidelity(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    output_indices: Sequence[int] = (0,),
    target_indices: Sequence[int] = (0,),
    target_negation: bool = False,
    stream: ResidualStream = "residual",
) -> ResidualGroups:
    """Evaluate direct supervised residuals or selected model outputs.

    Args:
        model_apply: Pure explicit-state model application callable.
        model_state: Differentiable model parameter PyTree.
        batch: Arrays containing `inputs`, `targets`, and optional `weight` and `projection` fields.
        output_indices: Model-output components compared with observations.
        target_indices: Target components aligned with `output_indices`.
        target_negation: Whether to negate targets before subtraction.
        stream: `"residual"` for direct supervised errors or `"output"` for selected model outputs.

    Returns:
        One residual or output group for scalar loss reduction or derivative-based balancing.

    Raises:
        ValueError: If the stream, component selection, or optional projection is invalid.
    """
    validate_stream(stream, supports_output=True)
    resolved_outputs = validate_component_indices(output_indices, option="output_indices")
    resolved_targets = validate_component_indices(target_indices, option="target_indices")
    projection = batch.get("projection")
    expected_target_count = 1 if projection is not None else len(resolved_outputs)
    if len(resolved_targets) != expected_target_count:
        raise ValueError("`target_indices` must select one projected target or one target per selected model output.")
    output = evaluate_selected_outputs(model_apply, model_state, batch["inputs"], resolved_outputs)
    if stream == "output":
        return ((output,),)
    target = select_components(batch["targets"], resolved_targets, option="target_indices")
    residual = base_data_fidelity_residual(
        output,
        target,
        weight=batch.get("weight"),
        projection=projection,
        target_negation=target_negation,
    )
    return ((residual,),)


def phase_wrapped_residuals(
    output: jax.Array,
    target: jax.Array,
    period: jax.Array,
    *,
    weight: jax.Array | None = None,
    target_negation: bool = False,
) -> tuple[jax.Array, jax.Array]:
    """Return cosine and sine residuals for periodic scalar observations.

    Args:
        output: Predicted scalar values.
        target: Observed scalar values.
        period: Sample-wise wrapping periods.
        weight: Optional multiplicative residual weights.
        target_negation: Whether to negate targets before phase conversion.

    Returns:
        Cosine and sine residual arrays with the broadcast input shape.
    """
    signed_target = -target if target_negation else target
    output_phase = jnp.pi * output / period
    target_phase = jnp.pi * signed_target / period
    residual_weight = 1.0 if weight is None else weight
    cosine = residual_weight * (jnp.cos(output_phase) - jnp.cos(target_phase))
    sine = residual_weight * (jnp.sin(output_phase) - jnp.sin(target_phase))
    return cosine, sine


@residual_equation(names=("phase",))
def phase_wrapped_fidelity(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    output_indices: Sequence[int] = (0,),
    target_indices: Sequence[int] = (0,),
    target_negation: bool = False,
    stream: ResidualStream = "residual",
) -> ResidualGroups:
    """Evaluate grouped phase-wrapped fidelity residuals or supervised outputs.

    Args:
        model_apply: Pure explicit-state model application callable.
        model_state: Differentiable model parameter PyTree.
        batch: Arrays containing `inputs`, `targets`, `period`, and optional `weight`.
        output_indices: Model-output components compared with observations.
        target_indices: Target components aligned with `output_indices`.
        target_negation: Whether to negate targets before phase conversion.
        stream: `"residual"` for cosine and sine errors or `"output"` for selected model outputs.

    Returns:
        One residual group containing cosine and sine arrays, or one selected-output group for output-based NTK
        diagnostics.

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
    cosine, sine = phase_wrapped_residuals(
        output,
        target,
        batch["period"],
        weight=batch.get("weight"),
        target_negation=target_negation,
    )
    return ((cosine, sine),)


__all__ = ["base_data_fidelity", "base_data_fidelity_residual", "phase_wrapped_fidelity", "phase_wrapped_residuals"]
