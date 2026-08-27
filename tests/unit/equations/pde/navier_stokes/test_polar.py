import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.equations.pde.navier_stokes import polar_navier_stokes


def test_zero_viscosity_polar_residuals_match_analytic_fields() -> None:
    """Verify all polar residual components against fields with closed-form derivatives."""

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Evaluate analytic velocity and pressure fields at one polar coordinate.

        Args:
            params: Unused placeholder for the generic explicit model-state contract.
            point: Coordinate vector ordered as `[r, th, t]`.

        Returns:
            Analytic output vector ordered as `[u_r, u_th, p]`.
        """
        del params
        radius, theta, time = point
        return jnp.stack((radius**2 + time, radius * theta, radius + theta * time))

    inputs = jnp.asarray([[0.5, 0.2, 0.3], [0.8, -0.1, 0.7]], dtype=jnp.float32)
    groups = polar_navier_stokes(predict_one, None, {"inputs": inputs})
    residuals = jnp.concatenate([group[0] for group in groups], axis=-1)
    radius, theta, time = np.asarray(inputs).T
    expected = np.stack(
        (
            3.0 * radius + time / radius + 1.0,
            2.0 + 2.0 * radius**3 + 2.0 * radius * time - radius * theta**2,
            2.0 * theta * (radius**2 + time) + radius * theta + time / radius,
        ),
        axis=-1,
    )
    np.testing.assert_allclose(np.asarray(residuals), expected, rtol=1e-5, atol=1e-6)


def test_viscous_polar_residuals_include_vector_laplacian_terms() -> None:
    """Verify both polar vector-Laplacian corrections against manufactured fields."""

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Evaluate manufactured velocity and pressure fields.

        Args:
            params: Unused placeholder for the generic explicit-state contract.
            point: Coordinate vector ordered as `[r, th, t]`.

        Returns:
            Polynomial output vector ordered as `[u_r, u_th, p]`.
        """
        del params
        radius, theta, time = point
        return jnp.stack((radius**2 + theta**2 + time, radius * theta + theta**2, radius + theta * time))

    inputs = jnp.asarray([[0.5, 0.2, 0.3], [0.8, -0.1, 0.7]], dtype=jnp.float32)
    viscosity_coefficient = 0.25
    inviscid_groups = polar_navier_stokes(predict_one, None, {"inputs": inputs})
    viscous_groups = polar_navier_stokes(
        predict_one,
        None,
        {"inputs": inputs},
        viscosity_coefficient=viscosity_coefficient,
    )
    inviscid = jnp.concatenate([group[0] for group in inviscid_groups], axis=-1)
    viscous = jnp.concatenate([group[0] for group in viscous_groups], axis=-1)
    radius, theta, time = np.asarray(inputs).T
    viscous_r = 3.0 + (2.0 - theta**2 - time - 2.0 * radius - 4.0 * theta) / radius**2
    viscous_theta = (2.0 - theta**2 + 4.0 * theta) / radius**2
    expected = np.asarray(inviscid).copy()
    expected[:, 1] -= viscosity_coefficient * viscous_r
    expected[:, 2] -= viscosity_coefficient * viscous_theta
    np.testing.assert_allclose(np.asarray(viscous), expected, rtol=1e-5, atol=1e-6)


def test_viscous_polar_residuals_have_finite_parameter_gradients() -> None:
    """Verify differentiation through viscous second derivatives remains finite."""

    def predict_one(params: jax.Array, point: jax.Array) -> jax.Array:
        """Evaluate parameterized manufactured fields.

        Args:
            params: Two trainable scalar coefficients.
            point: Coordinate vector ordered as `[r, th, t]`.

        Returns:
            Parameterized output vector ordered as `[u_r, u_th, p]`.
        """
        radius, theta, time = point
        return jnp.stack(
            (
                params[0] * (radius**2 + theta**2) + time,
                params[1] * (radius * theta + theta**2),
                radius + theta * time,
            )
        )

    inputs = jnp.asarray([[0.5, 0.2, 0.3], [0.8, -0.1, 0.7]], dtype=jnp.float32)

    def residual_loss(params: jax.Array) -> jax.Array:
        """Reduce viscous residuals to a differentiable scalar.

        Args:
            params: Two trainable scalar coefficients.

        Returns:
            Sum of squared viscous residual components.
        """
        groups = polar_navier_stokes(
            predict_one,
            params,
            {"inputs": inputs},
            viscosity_coefficient=0.2,
        )
        residuals = jnp.concatenate([group[0] for group in groups], axis=-1)
        return jnp.sum(residuals**2)

    gradients = jax.grad(residual_loss)(jnp.asarray([1.0, 0.5], dtype=jnp.float32))
    assert gradients.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(gradients)))


def test_polar_navier_stokes_returns_three_grouped_residual_components() -> None:
    """Verify the framework-facing equation separates continuity and both momentum components."""

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Evaluate a simple differentiable polar field.

        Args:
            params: Unused model-state placeholder.
            point: Coordinate vector ordered as `[r, th, t]`.

        Returns:
            Output vector ordered as `[u_r, u_th, p]`.
        """
        del params
        radius, theta, time = point
        return jnp.stack((radius + time, radius * theta, theta))

    batch = {"inputs": jnp.asarray([[0.5, 0.2, 0.3], [0.8, -0.1, 0.7]], dtype=jnp.float32)}
    groups = polar_navier_stokes(predict_one, None, batch)

    assert len(groups) == 3
    assert all(len(group) == 1 for group in groups)
    assert all(group[0].shape == (2, 1) for group in groups)


def test_polar_navier_stokes_rejects_output_stream_and_invalid_coefficients() -> None:
    """Verify the PDE exposes only residual streams and validates static physical settings."""

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Return an identity output for static policy validation.

        Args:
            params: Unused model-state placeholder.
            point: Three-component input point.

        Returns:
            Unchanged three-component point.
        """
        del params
        return point

    batch = {"inputs": jnp.ones((2, 3))}
    with pytest.raises(ValueError, match="does not expose"):
        polar_navier_stokes(predict_one, None, batch, stream="output")
    with pytest.raises(ValueError, match="viscosity_coefficient"):
        polar_navier_stokes(predict_one, None, batch, viscosity_coefficient=-0.1)
    with pytest.raises(ValueError, match="radius_epsilon"):
        polar_navier_stokes(predict_one, None, batch, radius_epsilon=0.0)
