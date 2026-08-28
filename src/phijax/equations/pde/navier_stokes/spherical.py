from typing import Any

import jax
import jax.numpy as jnp

from phijax.derivatives import hessian_diagonal, value_and_jacobian
from phijax.equations._common import validate_stream
from phijax.equations.metadata import residual_equation
from phijax.types import ArrayMapping, ModelApply, ResidualGroups, ResidualStream


@residual_equation(names=("continuity", "momentum_r", "momentum_th", "momentum_phi"))
def spherical_navier_stokes(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    pressure_coefficient: float = 1.0,
    viscosity_coefficient: float = 0.0,
    radius_epsilon: float = 1.0e-12,
    sine_epsilon: float = 1.0e-12,
    stream: ResidualStream = "residual",
) -> ResidualGroups:
    """Evaluate grouped weighted spherical Navier--Stokes residuals.

    The coordinate order is `[r, th, phi, t]`, model outputs are `[u_r, u_th, u_phi, p]`, and the returned groups
    represent continuity and radial, polar, and azimuthal momentum. Every physical residual is multiplied by
    `r sin(th)`. Away from the spherical axis this preserves its zero set, while removing reciprocal-sine factors from
    the inviscid equations and reducing their influence near the coordinate singularity.

    With scalar spherical Laplacian
    `L(f) = d2f/dr2 + (2 / r) df/dr + (1 / r**2) (d2f/dth2 + cot(th) df/dth
    + d2f/dphi2 / sin(th)**2)`, the unweighted viscous velocity terms are:

    - radial: `L(u_r) - 2 u_r/r**2 - (2 / r**2) (du_th/dth + cot(th) u_th)
      - 2 du_phi/dphi/(r**2 sin(th))`;
    - polar: `L(u_th) + 2 du_r/dth/r**2 - u_th/(r**2 sin(th)**2)
      - 2 cos(th) du_phi/dphi/(r**2 sin(th)**2)`; and
    - azimuthal: `L(u_phi) + 2 du_r/dphi/(r**2 sin(th))
      + 2 cos(th) du_th/dphi/(r**2 sin(th)**2) - u_phi/(r**2 sin(th)**2)`.

    A zero `viscosity_coefficient` selects the inviscid path and avoids tracing all second derivatives.

    Args:
        model_apply: Pure callable mapping `(model_state, point)` to `[u_r, u_th, u_phi, p]`.
        model_state: Explicit differentiable model state.
        batch: Arrays containing rank-two `[r, th, phi, t]` coordinates under `inputs`.
        pressure_coefficient: Coefficient multiplying spherical pressure gradients.
        viscosity_coefficient: Nonnegative coefficient multiplying spherical vector-Laplacian terms. Use `0.0` for
            inviscid flow or `1 / Re` for a nondimensional viscous equation.
        radius_epsilon: Minimum radius used in reciprocal geometric terms.
        sine_epsilon: Minimum value of `sin(th)` used in reciprocal geometric terms.
        stream: Equation representation requested by the objective. Navier--Stokes supports only `"residual"`.

    Returns:
        Four single-array groups ordered as continuity, radial momentum, polar momentum, and azimuthal momentum.

    Raises:
        ValueError: If `stream` requests raw model outputs or the equation settings are invalid.
    """
    validate_stream(stream, supports_output=False)
    inputs = batch["inputs"]
    if inputs.shape[-1] != 4:
        raise ValueError("Spherical PDE inputs must have columns `[r, th, phi, t]`.")
    if viscosity_coefficient < 0.0:
        raise ValueError("`viscosity_coefficient` must be nonnegative.")
    if radius_epsilon <= 0.0:
        raise ValueError("`radius_epsilon` must be positive.")
    if sine_epsilon <= 0.0:
        raise ValueError("`sine_epsilon` must be positive.")

    def vector_prediction(
        state: Any,
        radius: jax.Array,
        theta: jax.Array,
        phi: jax.Array,
        time_coordinate: jax.Array,
    ) -> jax.Array:
        """Evaluate all spherical fields from split coordinates.

        Args:
            state: Explicit differentiable model state.
            radius: Scalar radial coordinate.
            theta: Scalar polar angle.
            phi: Scalar azimuthal angle.
            time_coordinate: Scalar time coordinate.

        Returns:
            Model output vector ordered as `[u_r, u_th, u_phi, p]`.
        """
        return model_apply(state, jnp.stack((radius, theta, phi, time_coordinate)))

    def velocity_prediction(
        state: Any,
        radius: jax.Array,
        theta: jax.Array,
        phi: jax.Array,
        time_coordinate: jax.Array,
    ) -> jax.Array:
        """Evaluate only spherical velocity fields for second derivatives.

        Args:
            state: Explicit differentiable model state.
            radius: Scalar radial coordinate.
            theta: Scalar polar angle.
            phi: Scalar azimuthal angle.
            time_coordinate: Scalar time coordinate.

        Returns:
            Velocity vector ordered as `[u_r, u_th, u_phi]`.
        """
        return vector_prediction(state, radius, theta, phi, time_coordinate)[:3]

    value_and_first_derivatives = value_and_jacobian(vector_prediction, (1, 2, 3, 4))
    velocity_hessian_diagonal = (
        hessian_diagonal(velocity_prediction, (1, 2, 3)) if viscosity_coefficient != 0.0 else None
    )

    def point_residual(point: jax.Array) -> jax.Array:
        """Evaluate all weighted residual components at one spherical coordinate.

        Args:
            point: Coordinate vector ordered as `[r, th, phi, t]`.

        Returns:
            Residual vector ordered as `[continuity, momentum_r, momentum_th, momentum_phi]`.
        """
        prediction, (derivative_r, derivative_theta, derivative_phi, derivative_t) = value_and_first_derivatives(
            model_state,
            point[0],
            point[1],
            point[2],
            point[3],
        )
        radius = jnp.maximum(point[0], jnp.asarray(radius_epsilon, dtype=point.dtype))
        sin_theta = jnp.maximum(jnp.sin(point[1]), jnp.asarray(sine_epsilon, dtype=point.dtype))
        cos_theta = jnp.cos(point[1])
        r_sin_theta = radius * sin_theta
        u_r, u_theta, u_phi, _ = prediction
        du_r_dr, du_theta_dr, du_phi_dr, dp_dr = derivative_r
        du_r_dtheta, du_theta_dtheta, du_phi_dtheta, dp_dtheta = derivative_theta
        du_r_dphi, du_theta_dphi, du_phi_dphi, dp_dphi = derivative_phi
        du_r_dt, du_theta_dt, du_phi_dt, _ = derivative_t

        # Expanding the `r sin(th)` factor avoids singular divisions throughout the inviscid residuals.
        continuity = (
            r_sin_theta * du_r_dr
            + 2.0 * sin_theta * u_r
            + sin_theta * du_theta_dtheta
            + cos_theta * u_theta
            + du_phi_dphi
        )
        momentum_r = (
            r_sin_theta * du_r_dt
            + r_sin_theta * u_r * du_r_dr
            + sin_theta * u_theta * du_r_dtheta
            + u_phi * du_r_dphi
            - sin_theta * (u_theta**2 + u_phi**2)
            + pressure_coefficient * r_sin_theta * dp_dr
        )
        momentum_theta = (
            r_sin_theta * du_theta_dt
            + r_sin_theta * u_r * du_theta_dr
            + sin_theta * u_theta * du_theta_dtheta
            + u_phi * du_theta_dphi
            + sin_theta * u_r * u_theta
            - cos_theta * u_phi**2
            + pressure_coefficient * sin_theta * dp_dtheta
        )
        momentum_phi = (
            r_sin_theta * du_phi_dt
            + r_sin_theta * u_r * du_phi_dr
            + sin_theta * u_theta * du_phi_dtheta
            + u_phi * du_phi_dphi
            + sin_theta * u_r * u_phi
            + cos_theta * u_theta * u_phi
            + pressure_coefficient * dp_dphi
        )

        if velocity_hessian_diagonal is not None:
            second_r, second_theta, second_phi = velocity_hessian_diagonal(
                model_state,
                point[0],
                point[1],
                point[2],
                point[3],
            )
            inv_radius = 1.0 / radius
            inv_radius_squared = inv_radius**2
            inv_sin_theta = 1.0 / sin_theta
            cot_theta = cos_theta * inv_sin_theta
            velocity_r = derivative_r[:3]
            velocity_theta = derivative_theta[:3]
            scalar_laplacian = (
                second_r
                + 2.0 * inv_radius * velocity_r
                + inv_radius_squared * (second_theta + cot_theta * velocity_theta + inv_sin_theta**2 * second_phi)
            )
            weighted_viscous_r = r_sin_theta * scalar_laplacian[0] - 2.0 * inv_radius * (
                sin_theta * (u_r + du_theta_dtheta) + cos_theta * u_theta + du_phi_dphi
            )
            weighted_viscous_theta = (
                r_sin_theta * scalar_laplacian[1]
                + 2.0 * sin_theta * inv_radius * du_r_dtheta
                - inv_radius * inv_sin_theta * (u_theta + 2.0 * cos_theta * du_phi_dphi)
            )
            weighted_viscous_phi = (
                r_sin_theta * scalar_laplacian[2]
                + 2.0 * inv_radius * du_r_dphi
                + inv_radius * inv_sin_theta * (2.0 * cos_theta * du_theta_dphi - u_phi)
            )
            coefficient = jnp.asarray(viscosity_coefficient, dtype=point.dtype)
            momentum_r = momentum_r - coefficient * weighted_viscous_r
            momentum_theta = momentum_theta - coefficient * weighted_viscous_theta
            momentum_phi = momentum_phi - coefficient * weighted_viscous_phi
        return jnp.stack((continuity, momentum_r, momentum_theta, momentum_phi))

    residuals = jax.vmap(point_residual)(inputs)
    return tuple((residuals[..., index : index + 1],) for index in range(4))


__all__ = ["spherical_navier_stokes"]
