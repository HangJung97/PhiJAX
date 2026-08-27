import math
from typing import Any

import jax
import jax.numpy as jnp

from phijax.derivatives import hessian_diagonal, value_and_jacobian
from phijax.equations._common import validate_stream
from phijax.equations.metadata import residual_equation
from phijax.types import ArrayMapping, ModelApply, ResidualGroups, ResidualStream


@residual_equation(names=("burgers",))
def burgers_1d(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    viscosity_coefficient: float = 0.01 / math.pi,
    output_index: int = 0,
    stream: ResidualStream = "residual",
) -> ResidualGroups:
    """Evaluate a grouped one-dimensional Burgers equation residual.

    Coordinates are ordered as `[t, x]`. For the selected scalar model output `u`, the returned residual is
    `du/dt + u * du/dx - viscosity_coefficient * d2u/dx2`. A zero viscosity selects the inviscid path and avoids
    tracing a second spatial derivative. The viscous path computes only `d2u/dx2`, without tracing the unused mixed
    derivative `d2u/(dx dt)`.

    Args:
        model_apply: Pure callable mapping `(model_state, point)` to model outputs.
        model_state: Explicit differentiable model state.
        batch: Arrays containing rank-two `[t, x]` coordinates under `inputs`.
        viscosity_coefficient: Nonnegative coefficient multiplying `d2u/dx2`.
        output_index: Nonnegative scalar model-output component representing `u`.
        stream: Equation representation requested by the objective. Burgers supports only `"residual"`.

    Returns:
        One single-array residual group.

    Raises:
        ValueError: If `stream` requests raw model outputs or the equation settings are invalid.
    """
    validate_stream(stream, supports_output=False)
    inputs = batch["inputs"]
    if inputs.ndim != 2 or inputs.shape[-1] != 2:
        raise ValueError("Burgers PDE inputs must be rank two with columns `[t, x]`.")
    if viscosity_coefficient < 0.0:
        raise ValueError("`viscosity_coefficient` must be nonnegative.")
    if output_index < 0:
        raise ValueError("`output_index` must be nonnegative.")

    def scalar_prediction(
        state: Any,
        time_coordinate: jax.Array,
        position: jax.Array,
    ) -> jax.Array:
        """Evaluate the selected scalar output from split coordinates.

        Args:
            state: Explicit differentiable model state.
            time_coordinate: Scalar time coordinate.
            position: Scalar spatial coordinate.

        Returns:
            Selected scalar model output.
        """
        point = jnp.stack((time_coordinate, position))
        return model_apply(state, point)[output_index]

    value_and_first_derivatives = value_and_jacobian(scalar_prediction, (1, 2))
    second_spatial_derivative = hessian_diagonal(scalar_prediction, 2) if viscosity_coefficient != 0.0 else None

    def point_residual(point: jax.Array) -> jax.Array:
        """Evaluate the scalar Burgers residual at one physical coordinate.

        Args:
            point: Coordinate vector ordered as `[t, x]`.

        Returns:
            Scalar equation residual.
        """
        value, (du_dt, du_dx) = value_and_first_derivatives(model_state, point[0], point[1])
        residual = du_dt + value * du_dx
        if second_spatial_derivative is not None:
            (d2u_dx2,) = second_spatial_derivative(model_state, point[0], point[1])
            coefficient = jnp.asarray(viscosity_coefficient, dtype=value.dtype)
            residual = residual - coefficient * d2u_dx2
        return residual

    residuals = jax.vmap(point_residual)(inputs)[:, None]
    return ((residuals,),)


__all__ = ["burgers_1d"]
