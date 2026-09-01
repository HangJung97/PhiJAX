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
    r"""Evaluate grouped weighted spherical Navier--Stokes residuals.

    The coordinate order is `[r, th, phi, t]`, model outputs are `[u_r, u_th, u_phi, p]`, and the returned groups
    represent continuity and radial, polar, and azimuthal momentum. Every physical residual is multiplied by
    `r sin(th)`. Away from the spherical axis this preserves its zero set, while removing reciprocal-sine factors from
    the inviscid equations and reducing their influence near the coordinate singularity.

    With $c_p$ equal to `pressure_coefficient`, define the unweighted continuity and nonviscous momentum terms as

    $$
    \begin{aligned}
    C
        &= \frac{\partial u_r}{\partial r}
        + \frac{2u_r}{r}
        + \frac{1}{r}\frac{\partial u_\theta}{\partial \theta}
        + \frac{u_\theta\cot\theta}{r}
        + \frac{1}{r\sin\theta}\frac{\partial u_\phi}{\partial \phi}, \\
    F_r
        &= \frac{\partial u_r}{\partial t}
        + u_r\frac{\partial u_r}{\partial r}
        + \frac{u_\theta}{r}\frac{\partial u_r}{\partial \theta}
        + \frac{u_\phi}{r\sin\theta}\frac{\partial u_r}{\partial \phi}
        - \frac{u_\theta^2 + u_\phi^2}{r}
        + c_p\frac{\partial p}{\partial r}, \\
    F_\theta
        &= \frac{\partial u_\theta}{\partial t}
        + u_r\frac{\partial u_\theta}{\partial r}
        + \frac{u_\theta}{r}\frac{\partial u_\theta}{\partial \theta}
        + \frac{u_\phi}{r\sin\theta}\frac{\partial u_\theta}{\partial \phi}
        + \frac{u_ru_\theta}{r}
        - \frac{u_\phi^2\cot\theta}{r}
        + \frac{c_p}{r}\frac{\partial p}{\partial \theta}, \\
    F_\phi
        &= \frac{\partial u_\phi}{\partial t}
        + u_r\frac{\partial u_\phi}{\partial r}
        + \frac{u_\theta}{r}\frac{\partial u_\phi}{\partial \theta}
        + \frac{u_\phi}{r\sin\theta}\frac{\partial u_\phi}{\partial \phi}
        + \frac{u_ru_\phi}{r}
        + \frac{u_\theta u_\phi\cot\theta}{r}
        + \frac{c_p}{r\sin\theta}\frac{\partial p}{\partial \phi}.
    \end{aligned}
    $$

    Define the scalar spherical Laplacian as

    $$
    \mathcal{L}(f)
    = \frac{\partial^2 f}{\partial r^2}
    + \frac{2}{r}\frac{\partial f}{\partial r}
    + \frac{1}{r^2}\left(
        \frac{\partial^2 f}{\partial \theta^2}
        + \cot\theta\frac{\partial f}{\partial \theta}
        + \frac{1}{\sin^2\theta}\frac{\partial^2 f}{\partial \phi^2}
    \right).
    $$

    The unweighted viscous velocity terms are

    $$
    \begin{aligned}
    (\nabla^2\mathbf{u})_r
        &= \mathcal{L}(u_r)
        - \frac{2u_r}{r^2}
        - \frac{2}{r^2}\left(
            \frac{\partial u_\theta}{\partial \theta}
            + u_\theta\cot\theta
        \right)
        - \frac{2}{r^2\sin\theta}\frac{\partial u_\phi}{\partial \phi}, \\
    (\nabla^2\mathbf{u})_\theta
        &= \mathcal{L}(u_\theta)
        + \frac{2}{r^2}\frac{\partial u_r}{\partial \theta}
        - \frac{u_\theta}{r^2\sin^2\theta}
        - \frac{2\cos\theta}{r^2\sin^2\theta}
            \frac{\partial u_\phi}{\partial \phi}, \\
    (\nabla^2\mathbf{u})_\phi
        &= \mathcal{L}(u_\phi)
        + \frac{2}{r^2\sin\theta}\frac{\partial u_r}{\partial \phi}
        + \frac{2\cos\theta}{r^2\sin^2\theta}
            \frac{\partial u_\theta}{\partial \phi}
        - \frac{u_\phi}{r^2\sin^2\theta}.
    \end{aligned}
    $$

    With $c_\nu$ equal to `viscosity_coefficient`, the returned weighted residuals are

    $$
    R_c = r\sin\theta\,C,
    \qquad
    R_r = r\sin\theta\left[F_r - c_\nu(\nabla^2\mathbf{u})_r\right],
    \qquad
    R_\theta = r\sin\theta\left[F_\theta - c_\nu(\nabla^2\mathbf{u})_\theta\right],
    \qquad
    R_\phi = r\sin\theta\left[F_\phi - c_\nu(\nabla^2\mathbf{u})_\phi\right].
    $$

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
