from typing import Any

import jax
import jax.numpy as jnp

from phijax.derivatives import hessian_diagonal, value_and_jacobian
from phijax.equations._common import validate_stream
from phijax.equations.metadata import residual_equation
from phijax.types import ArrayMapping, ModelApply, ResidualGroups, ResidualStream


def _cartesian_navier_stokes_residuals(
    model_apply: ModelApply,
    model_state: Any,
    inputs: jax.Array,
    *,
    spatial_dimension: int,
    pressure_coefficient: float,
    viscosity_coefficient: float,
) -> jax.Array:
    """Evaluate Cartesian incompressible Navier--Stokes residuals in a fixed dimension.

    Args:
        model_apply: Pure callable mapping `(model_state, point)` to velocity components followed by pressure.
        model_state: Explicit differentiable model state.
        inputs: Rank-two coordinates with spatial columns followed by time.
        spatial_dimension: Number of Cartesian spatial coordinates and velocity components.
        pressure_coefficient: Coefficient multiplying pressure gradients.
        viscosity_coefficient: Nonnegative coefficient multiplying velocity Laplacians.

    Returns:
        Residual matrix containing continuity followed by one momentum component per spatial coordinate.
    """

    def vector_prediction(state: Any, *coordinates: jax.Array) -> jax.Array:
        """Evaluate all Cartesian fields from split scalar coordinates.

        Args:
            state: Explicit differentiable model state.
            *coordinates: Spatial coordinates followed by time.

        Returns:
            Velocity components followed by pressure.
        """
        return model_apply(state, jnp.stack(coordinates))

    def velocity_prediction(state: Any, *coordinates: jax.Array) -> jax.Array:
        """Evaluate only Cartesian velocity components for second derivatives.

        Args:
            state: Explicit differentiable model state.
            *coordinates: Spatial coordinates followed by time.

        Returns:
            Velocity vector with `spatial_dimension` components.
        """
        return vector_prediction(state, *coordinates)[:spatial_dimension]

    coordinate_argnums = tuple(range(1, spatial_dimension + 2))
    spatial_argnums = coordinate_argnums[:-1]
    value_and_first_derivatives = value_and_jacobian(vector_prediction, coordinate_argnums)
    velocity_hessian_diagonal = (
        hessian_diagonal(velocity_prediction, spatial_argnums) if viscosity_coefficient != 0.0 else None
    )

    def point_residual(point: jax.Array) -> jax.Array:
        """Evaluate continuity and momentum residuals at one Cartesian point.

        Args:
            point: Spatial coordinates followed by time.

        Returns:
            Continuity followed by Cartesian momentum residuals.
        """
        prediction, first_derivatives = value_and_first_derivatives(model_state, *point)
        if prediction.shape != (spatial_dimension + 1,):
            expected_fields = f"{spatial_dimension} velocity components and pressure"
            raise ValueError(f"Cartesian {spatial_dimension}D models must return {expected_fields}.")
        spatial_derivatives = first_derivatives[:-1]
        time_derivative = first_derivatives[-1]
        velocity = prediction[:spatial_dimension]

        continuity = sum(spatial_derivatives[axis][axis] for axis in range(spatial_dimension))
        momentum = []
        for component in range(spatial_dimension):
            convection = sum(velocity[axis] * spatial_derivatives[axis][component] for axis in range(spatial_dimension))
            pressure_gradient = spatial_derivatives[component][spatial_dimension]
            component_residual = time_derivative[component] + convection + pressure_coefficient * pressure_gradient
            momentum.append(component_residual)

        if velocity_hessian_diagonal is not None:
            second_derivatives = velocity_hessian_diagonal(model_state, *point)
            coefficient = jnp.asarray(viscosity_coefficient, dtype=point.dtype)
            momentum = [
                residual - coefficient * sum(second_derivatives[axis][component] for axis in range(spatial_dimension))
                for component, residual in enumerate(momentum)
            ]
        return jnp.stack((continuity, *momentum))

    return jax.vmap(point_residual)(inputs)


@residual_equation(names=("continuity", "momentum_x", "momentum_y"))
def cartesian_2d_navier_stokes(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    pressure_coefficient: float = 1.0,
    viscosity_coefficient: float = 0.0,
    stream: ResidualStream = "residual",
) -> ResidualGroups:
    """Evaluate grouped incompressible Navier--Stokes residuals in two Cartesian dimensions.

    The coordinate order is `[x, y, t]`, model outputs are `[u_x, u_y, p]`, and the returned groups represent
    continuity, x momentum, and y momentum. With `c_p=pressure_coefficient` and `c_nu=viscosity_coefficient`, the
    residuals are:

    - continuity: `du_x/dx + du_y/dy`;
    - x momentum: `du_x/dt + u_x du_x/dx + u_y du_x/dy + c_p dp/dx
      - c_nu (d2u_x/dx2 + d2u_x/dy2)`; and
    - y momentum: `du_y/dt + u_x du_y/dx + u_y du_y/dy + c_p dp/dy
      - c_nu (d2u_y/dx2 + d2u_y/dy2)`.

    A zero `viscosity_coefficient` selects the inviscid path and avoids tracing second derivatives.

    Args:
        model_apply: Pure callable mapping `(model_state, point)` to `[u_x, u_y, p]`.
        model_state: Explicit differentiable model state.
        batch: Arrays containing rank-two physical coordinates under `inputs`, with columns `[x, y, t]`.
        pressure_coefficient: Coefficient multiplying Cartesian pressure gradients.
        viscosity_coefficient: Nonnegative coefficient multiplying Cartesian velocity Laplacians. Use `0.0` for
            inviscid flow or `1 / Re` for a nondimensional viscous equation.
        stream: Equation representation requested by the objective. Navier--Stokes supports only `"residual"`.

    Returns:
        Three single-array groups ordered as continuity, x momentum, and y momentum.

    Raises:
        ValueError: If `stream` requests model outputs or the equation settings are invalid.
    """
    validate_stream(stream, supports_output=False)
    inputs = batch["inputs"]
    if inputs.shape[-1] != 3:
        raise ValueError("Cartesian 2D PDE inputs must have columns `[x, y, t]`.")
    if viscosity_coefficient < 0.0:
        raise ValueError("`viscosity_coefficient` must be nonnegative.")
    residuals = _cartesian_navier_stokes_residuals(
        model_apply,
        model_state,
        inputs,
        spatial_dimension=2,
        pressure_coefficient=pressure_coefficient,
        viscosity_coefficient=viscosity_coefficient,
    )
    return tuple((residuals[..., index : index + 1],) for index in range(3))


@residual_equation(names=("continuity", "momentum_x", "momentum_y", "momentum_z"))
def cartesian_3d_navier_stokes(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    pressure_coefficient: float = 1.0,
    viscosity_coefficient: float = 0.0,
    stream: ResidualStream = "residual",
) -> ResidualGroups:
    """Evaluate grouped incompressible Navier--Stokes residuals in three Cartesian dimensions.

    The coordinate order is `[x, y, z, t]`, model outputs are `[u_x, u_y, u_z, p]`, and the returned groups represent
    continuity and the x, y, and z momentum equations. With `c_p=pressure_coefficient` and
    `c_nu=viscosity_coefficient`, the residuals are:

    - continuity: `du_x/dx + du_y/dy + du_z/dz`;
    - x momentum: `du_x/dt + u_x du_x/dx + u_y du_x/dy + u_z du_x/dz + c_p dp/dx
      - c_nu (d2u_x/dx2 + d2u_x/dy2 + d2u_x/dz2)`;
    - y momentum: `du_y/dt + u_x du_y/dx + u_y du_y/dy + u_z du_y/dz + c_p dp/dy
      - c_nu (d2u_y/dx2 + d2u_y/dy2 + d2u_y/dz2)`; and
    - z momentum: `du_z/dt + u_x du_z/dx + u_y du_z/dy + u_z du_z/dz + c_p dp/dz
      - c_nu (d2u_z/dx2 + d2u_z/dy2 + d2u_z/dz2)`.

    A zero `viscosity_coefficient` selects the inviscid path and avoids tracing second derivatives.

    Args:
        model_apply: Pure callable mapping `(model_state, point)` to `[u_x, u_y, u_z, p]`.
        model_state: Explicit differentiable model state.
        batch: Arrays containing rank-two physical coordinates under `inputs`, with columns `[x, y, z, t]`.
        pressure_coefficient: Coefficient multiplying Cartesian pressure gradients.
        viscosity_coefficient: Nonnegative coefficient multiplying Cartesian velocity Laplacians. Use `0.0` for
            inviscid flow or `1 / Re` for a nondimensional viscous equation.
        stream: Equation representation requested by the objective. Navier--Stokes supports only `"residual"`.

    Returns:
        Four single-array groups ordered as continuity, x momentum, y momentum, and z momentum.

    Raises:
        ValueError: If `stream` requests model outputs or the equation settings are invalid.
    """
    validate_stream(stream, supports_output=False)
    inputs = batch["inputs"]
    if inputs.shape[-1] != 4:
        raise ValueError("Cartesian 3D PDE inputs must have columns `[x, y, z, t]`.")
    if viscosity_coefficient < 0.0:
        raise ValueError("`viscosity_coefficient` must be nonnegative.")
    residuals = _cartesian_navier_stokes_residuals(
        model_apply,
        model_state,
        inputs,
        spatial_dimension=3,
        pressure_coefficient=pressure_coefficient,
        viscosity_coefficient=viscosity_coefficient,
    )
    return tuple((residuals[..., index : index + 1],) for index in range(4))


__all__ = ["cartesian_2d_navier_stokes", "cartesian_3d_navier_stokes"]
