from typing import Any

import jax
import jax.numpy as jnp

from phijax.derivatives import hessian_diagonal, value_and_jacobian
from phijax.equations._common import validate_stream
from phijax.equations.metadata import residual_equation
from phijax.types import ArrayMapping, ModelApply, ResidualGroups, ResidualStream


@residual_equation(names=("continuity", "momentum_r", "momentum_th"))
def polar_navier_stokes(
    model_apply: ModelApply,
    model_state: Any,
    batch: ArrayMapping,
    *,
    pressure_coefficient: float = 1.0,
    viscosity_coefficient: float = 0.0,
    radius_epsilon: float = 1.0e-12,
    stream: ResidualStream = "residual",
) -> ResidualGroups:
    r"""Evaluate grouped polar Navier--Stokes equation residuals.

    The coordinate order is `[r, th, t]`, and model outputs are `[u_r, u_th, p]`. The returned groups represent
    continuity, radial momentum, and angular momentum. All reciprocal-radius terms use `max(r, radius_epsilon)`. A zero
    `viscosity_coefficient` selects the inviscid path and avoids tracing second derivatives. A nonzero coefficient adds
    the polar vector-Laplacian terms used by the viscous incompressible Navier--Stokes equations. The viscous path
    computes only pure radial and angular second derivatives of velocity, excluding unused mixed and pressure Hessian
    entries.

    With $c_p$ equal to `pressure_coefficient`, define the continuity and nonviscous momentum terms as

    $$
    \begin{aligned}
    C
        &= \frac{\partial u_r}{\partial r}
        + \frac{u_r}{r}
        + \frac{1}{r}\frac{\partial u_\theta}{\partial \theta}, \\
    F_r
        &= \frac{\partial u_r}{\partial t}
        + u_r\frac{\partial u_r}{\partial r}
        + \frac{u_\theta}{r}\frac{\partial u_r}{\partial \theta}
        - \frac{u_\theta^2}{r}
        + c_p\frac{\partial p}{\partial r}, \\
    F_\theta
        &= \frac{\partial u_\theta}{\partial t}
        + u_r\frac{\partial u_\theta}{\partial r}
        + \frac{u_\theta}{r}\frac{\partial u_\theta}{\partial \theta}
        + \frac{u_ru_\theta}{r}
        + \frac{c_p}{r}\frac{\partial p}{\partial \theta}.
    \end{aligned}
    $$

    Define the scalar polar Laplacian as

    $$
    \mathcal{L}(f)
    = \frac{\partial^2 f}{\partial r^2}
    + \frac{1}{r}\frac{\partial f}{\partial r}
    + \frac{1}{r^2}\frac{\partial^2 f}{\partial \theta^2}.
    $$

    The viscous velocity terms are

    $$
    \begin{aligned}
    (\nabla^2\mathbf{u})_r
        &= \mathcal{L}(u_r)
        - \frac{u_r}{r^2}
        - \frac{2}{r^2}\frac{\partial u_\theta}{\partial \theta}, \\
    (\nabla^2\mathbf{u})_\theta
        &= \mathcal{L}(u_\theta)
        - \frac{u_\theta}{r^2}
        + \frac{2}{r^2}\frac{\partial u_r}{\partial \theta}.
    \end{aligned}
    $$

    With $c_\nu$ equal to `viscosity_coefficient`, the returned residuals are

    $$
    R_c = C,
    \qquad
    R_r = F_r - c_\nu(\nabla^2\mathbf{u})_r,
    \qquad
    R_\theta = F_\theta - c_\nu(\nabla^2\mathbf{u})_\theta.
    $$

    Args:
        model_apply: Pure callable mapping `(model_state, point)` to `[u_r, u_th, p]`.
        model_state: Explicit differentiable model state.
        batch: Arrays containing rank-two `[r, th, t]` coordinates under `inputs`.
        pressure_coefficient: Coefficient multiplying radial and angular pressure gradients.
        viscosity_coefficient: Coefficient multiplying the polar vector-Laplacian terms. Use `0.0` for inviscid flow or
            `1 / Re` for a nondimensional viscous equation. This value must be static while tracing.
        radius_epsilon: Minimum radius used in reciprocal geometric terms.
        stream: Equation representation requested by the objective. Polar Navier--Stokes supports only `"residual"`.

    Returns:
        Three single-array groups ordered as continuity, radial momentum, and angular momentum.

    Raises:
        ValueError: If `stream` requests raw model outputs or the equation settings are invalid.
    """
    validate_stream(stream, supports_output=False)
    inputs = batch["inputs"]
    if inputs.shape[-1] != 3:
        raise ValueError("Polar PDE inputs must have columns `[r, th, t]`.")
    if viscosity_coefficient < 0.0:
        raise ValueError("`viscosity_coefficient` must be nonnegative.")
    if radius_epsilon <= 0.0:
        raise ValueError("`radius_epsilon` must be positive.")

    def vector_prediction(
        state: Any,
        radius: jax.Array,
        theta: jax.Array,
        time_coordinate: jax.Array,
    ) -> jax.Array:
        """Evaluate all polar fields from split coordinates.

        Args:
            state: Explicit differentiable model state.
            radius: Scalar radial coordinate.
            theta: Scalar angular coordinate.
            time_coordinate: Scalar time coordinate.

        Returns:
            Model output vector ordered as `[u_r, u_th, p]`.
        """
        return model_apply(state, jnp.stack((radius, theta, time_coordinate)))

    def velocity_prediction(
        state: Any,
        radius: jax.Array,
        theta: jax.Array,
        time_coordinate: jax.Array,
    ) -> jax.Array:
        """Evaluate only polar velocity fields for second derivatives.

        Args:
            state: Explicit differentiable model state.
            radius: Scalar radial coordinate.
            theta: Scalar angular coordinate.
            time_coordinate: Scalar time coordinate.

        Returns:
            Velocity vector ordered as `[u_r, u_th]`.
        """
        return vector_prediction(state, radius, theta, time_coordinate)[:2]

    value_and_first_derivatives = value_and_jacobian(vector_prediction, (1, 2, 3))
    velocity_hessian_diagonal = hessian_diagonal(velocity_prediction, (1, 2)) if viscosity_coefficient != 0.0 else None

    def point_residual(point: jax.Array) -> jax.Array:
        """Evaluate all residual components at one physical coordinate.

        Args:
            point: Coordinate vector ordered as `[r, th, t]`.

        Returns:
            Residual vector ordered as `[continuity, momentum_r, momentum_th]`.
        """
        prediction, (derivative_r, derivative_theta, derivative_t) = value_and_first_derivatives(
            model_state,
            point[0],
            point[1],
            point[2],
        )
        radius = jnp.maximum(point[0], jnp.asarray(radius_epsilon, dtype=point.dtype))
        inv_radius = 1.0 / radius
        inv_radius_squared = inv_radius**2
        u_r, u_theta, _ = prediction
        du_r_dr, du_theta_dr, dp_dr = derivative_r
        du_r_dtheta, du_theta_dtheta, dp_dtheta = derivative_theta
        du_r_dt, du_theta_dt, _ = derivative_t

        continuity = du_r_dr + u_r * inv_radius + du_theta_dtheta * inv_radius
        momentum_r = (
            du_r_dt
            + u_r * du_r_dr
            + u_theta * inv_radius * du_r_dtheta
            - u_theta**2 * inv_radius
            + pressure_coefficient * dp_dr
        )
        momentum_theta = (
            du_theta_dt
            + u_r * du_theta_dr
            + u_theta * inv_radius * du_theta_dtheta
            + u_r * u_theta * inv_radius
            + pressure_coefficient * dp_dtheta * inv_radius
        )
        if velocity_hessian_diagonal is not None:
            second_r, second_theta = velocity_hessian_diagonal(model_state, point[0], point[1], point[2])
            laplace_u_r = second_r[0] + du_r_dr * inv_radius + second_theta[0] * inv_radius_squared
            laplace_u_theta = second_r[1] + du_theta_dr * inv_radius + second_theta[1] * inv_radius_squared
            viscous_r = laplace_u_r - u_r * inv_radius_squared - 2.0 * du_theta_dtheta * inv_radius_squared
            viscous_theta = laplace_u_theta - u_theta * inv_radius_squared + 2.0 * du_r_dtheta * inv_radius_squared
            coefficient = jnp.asarray(viscosity_coefficient, dtype=point.dtype)
            momentum_r = momentum_r - coefficient * viscous_r
            momentum_theta = momentum_theta - coefficient * viscous_theta
        return jnp.stack((continuity, momentum_r, momentum_theta))

    residuals = jax.vmap(point_residual)(inputs)
    return (
        (residuals[..., 0:1],),
        (residuals[..., 1:2],),
        (residuals[..., 2:3],),
    )


__all__ = ["polar_navier_stokes"]
