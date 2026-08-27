import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.equations.pde.navier_stokes import spherical_navier_stokes


def test_inviscid_spherical_residuals_match_manufactured_fields() -> None:
    """Verify weighted continuity and all inviscid spherical momentum terms analytically."""

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Evaluate manufactured spherical velocity and pressure fields.

        Args:
            params: Unused explicit-state placeholder.
            point: Coordinate vector ordered as `[r, th, phi, t]`.

        Returns:
            Output vector ordered as `[u_r, u_th, u_phi, p]`.
        """
        del params
        radius, theta, phi, time = point
        return jnp.stack(
            (
                radius**2 + time,
                radius * theta,
                radius * phi + theta,
                radius + theta * time + phi**2,
            )
        )

    inputs = jnp.asarray([[0.7, 0.6, 0.3, 0.2], [1.5, 1.2, -0.2, 0.4]], dtype=jnp.float32)
    pressure_coefficient = 0.5
    groups = spherical_navier_stokes(
        predict_one,
        None,
        {"inputs": inputs},
        pressure_coefficient=pressure_coefficient,
    )
    residuals = jnp.concatenate([group[0] for group in groups], axis=-1)
    radius, theta, phi, time = np.asarray(inputs).T
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    u_r = radius**2 + time
    u_theta = radius * theta
    u_phi = radius * phi + theta
    expected = np.stack(
        (
            4.0 * radius**2 * sin_theta
            + 2.0 * time * sin_theta
            + radius * sin_theta
            + radius * theta * cos_theta
            + radius,
            radius * sin_theta
            + 2.0 * radius**2 * sin_theta * u_r
            - sin_theta * (u_theta**2 + u_phi**2)
            + pressure_coefficient * radius * sin_theta,
            radius * sin_theta * u_r * theta
            + radius * sin_theta * u_theta
            + sin_theta * u_r * u_theta
            - cos_theta * u_phi**2
            + pressure_coefficient * sin_theta * time,
            radius * sin_theta * u_r * phi
            + sin_theta * u_theta
            + radius * u_phi
            + sin_theta * u_r * u_phi
            + cos_theta * u_theta * u_phi
            + 2.0 * pressure_coefficient * phi,
        ),
        axis=-1,
    )

    assert residuals.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(residuals), expected, rtol=1e-5, atol=1e-6)


def test_viscous_spherical_residuals_include_vector_laplacian_corrections() -> None:
    """Verify weighted spherical vector-Laplacian corrections for constant azimuthal flow."""

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Evaluate constant azimuthal flow connected to the input coordinates.

        Args:
            params: Unused explicit-state placeholder.
            point: Coordinate vector ordered as `[r, th, phi, t]`.

        Returns:
            Output vector `[0, 0, 1, 0]` connected to `point`.
        """
        del params
        connected_zero = point[0] * 0.0
        return jnp.stack((connected_zero, connected_zero, connected_zero + 1.0, connected_zero))

    inputs = jnp.asarray([[0.7, 0.6, 0.3, 0.0], [1.5, 1.2, 0.8, 0.4]], dtype=jnp.float32)
    groups = spherical_navier_stokes(
        predict_one,
        None,
        {"inputs": inputs},
        viscosity_coefficient=1.0,
    )
    residuals = jnp.concatenate([group[0] for group in groups], axis=-1)
    radius, theta, _, _ = np.asarray(inputs).T
    expected = np.stack(
        (
            np.zeros_like(radius),
            -np.sin(theta),
            -np.cos(theta),
            1.0 / (radius * np.sin(theta)),
        ),
        axis=-1,
    )
    np.testing.assert_allclose(np.asarray(residuals), expected, rtol=1e-5, atol=1e-6)


def test_spherical_residuals_have_finite_parameter_gradients_and_axis_values() -> None:
    """Verify viscous parameter gradients and protected geometric denominators remain finite."""

    def predict_one(params: jax.Array, point: jax.Array) -> jax.Array:
        """Evaluate parameterized quadratic spherical velocity fields.

        Args:
            params: Three trainable velocity coefficients.
            point: Coordinate vector ordered as `[r, th, phi, t]`.

        Returns:
            Output vector ordered as `[u_r, u_th, u_phi, p]`.
        """
        radius, theta, phi, time = point
        return jnp.stack(
            (
                params[0] * radius**2 + time,
                params[1] * theta**2,
                params[2] * phi**2,
                radius + theta + phi,
            )
        )

    inputs = jnp.asarray([[0.0, 0.0, 0.2, 0.1], [0.8, 0.7, -0.1, 0.3]], dtype=jnp.float32)

    def residual_loss(params: jax.Array) -> jax.Array:
        """Reduce weighted viscous residuals to a differentiable scalar.

        Args:
            params: Three trainable velocity coefficients.

        Returns:
            Mean squared spherical residual.
        """
        groups = spherical_navier_stokes(
            predict_one,
            params,
            {"inputs": inputs},
            viscosity_coefficient=0.1,
            radius_epsilon=1.0e-3,
            sine_epsilon=1.0e-3,
        )
        residuals = jnp.concatenate([group[0] for group in groups], axis=-1)
        return jnp.mean(residuals**2)

    params = jnp.asarray([1.0, 0.5, -0.25], dtype=jnp.float32)
    groups = spherical_navier_stokes(
        predict_one,
        params,
        {"inputs": inputs},
        viscosity_coefficient=0.1,
        radius_epsilon=1.0e-3,
        sine_epsilon=1.0e-3,
    )
    residuals = jnp.concatenate([group[0] for group in groups], axis=-1)
    gradients = jax.grad(residual_loss)(params)
    assert residuals.shape == (2, 4)
    assert bool(jnp.all(jnp.isfinite(residuals)))
    assert bool(jnp.all(jnp.isfinite(gradients)))


def test_spherical_equation_returns_four_named_groups() -> None:
    """Verify the framework-facing spherical equation separates all four residual components."""

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Return connected zero spherical fields.

        Args:
            params: Unused explicit-state placeholder.
            point: Spherical coordinate vector.

        Returns:
            Four zero fields connected to `point`.
        """
        del params
        return jnp.zeros((4,), dtype=point.dtype) + point[0] * 0.0

    batch = {"inputs": jnp.asarray([[0.7, 0.6, 0.3, 0.0]], dtype=jnp.float32)}
    groups = spherical_navier_stokes(predict_one, None, batch)

    assert len(groups) == 4
    assert all(len(group) == 1 and group[0].shape == (1, 1) for group in groups)


def test_spherical_equation_rejects_invalid_settings_and_output_stream() -> None:
    """Verify spherical equations validate coordinate geometry, viscosity, and stream selection."""

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Return connected zero spherical fields.

        Args:
            params: Unused explicit-state placeholder.
            point: Spherical coordinate vector.

        Returns:
            Four zero fields connected to `point`.
        """
        del params
        return jnp.zeros((4,), dtype=point.dtype) + point[0] * 0.0

    inputs = jnp.ones((2, 4), dtype=jnp.float32)
    with pytest.raises(ValueError, match="does not expose"):
        spherical_navier_stokes(predict_one, None, {"inputs": inputs}, stream="output")
    with pytest.raises(ValueError, match="Spherical"):
        spherical_navier_stokes(predict_one, None, {"inputs": jnp.ones((2, 3))})
    with pytest.raises(ValueError, match="viscosity_coefficient"):
        spherical_navier_stokes(predict_one, None, {"inputs": inputs}, viscosity_coefficient=-0.1)
    with pytest.raises(ValueError, match="radius_epsilon"):
        spherical_navier_stokes(predict_one, None, {"inputs": inputs}, radius_epsilon=0.0)
    with pytest.raises(ValueError, match="sine_epsilon"):
        spherical_navier_stokes(predict_one, None, {"inputs": inputs}, sine_epsilon=0.0)
