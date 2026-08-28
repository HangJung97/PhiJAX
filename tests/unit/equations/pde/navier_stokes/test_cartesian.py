from collections.abc import Callable
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.equations.pde.navier_stokes import (
    cartesian_2d_navier_stokes,
    cartesian_3d_navier_stokes,
)


def test_cartesian_2d_residuals_match_manufactured_viscous_fields() -> None:
    """Verify two-dimensional continuity, convection, pressure, and diffusion terms analytically."""

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Evaluate manufactured two-dimensional velocity and pressure fields.

        Args:
            params: Unused explicit-state placeholder.
            point: Coordinate vector ordered as `[x, y, t]`.

        Returns:
            Output vector ordered as `[u_x, u_y, p]`.
        """
        del params
        x, y, time = point
        return jnp.stack((x**2 + y + time, x * y + y**2, x + y * time))

    inputs = jnp.asarray([[0.2, 0.3, 0.4], [0.7, -0.1, 0.8]], dtype=jnp.float32)
    pressure_coefficient = 0.5
    viscosity_coefficient = 0.2
    groups = cartesian_2d_navier_stokes(
        predict_one,
        None,
        {"inputs": inputs},
        pressure_coefficient=pressure_coefficient,
        viscosity_coefficient=viscosity_coefficient,
    )
    residuals = jnp.concatenate([group[0] for group in groups], axis=-1)
    x, y, time = np.asarray(inputs).T
    u_x = x**2 + y + time
    u_y = x * y + y**2
    expected = np.stack(
        (
            3.0 * x + 2.0 * y,
            1.0 + 2.0 * x * u_x + u_y + pressure_coefficient - 2.0 * viscosity_coefficient,
            y * u_x + (x + 2.0 * y) * u_y + pressure_coefficient * time - 2.0 * viscosity_coefficient,
        ),
        axis=-1,
    )

    assert residuals.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(residuals), expected, rtol=1e-5, atol=1e-6)


def test_cartesian_3d_residuals_match_manufactured_viscous_fields() -> None:
    """Verify three-dimensional continuity and all momentum components analytically."""

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Evaluate manufactured three-dimensional velocity and pressure fields.

        Args:
            params: Unused explicit-state placeholder.
            point: Coordinate vector ordered as `[x, y, z, t]`.

        Returns:
            Output vector ordered as `[u_x, u_y, u_z, p]`.
        """
        del params
        x, y, z, time = point
        return jnp.stack((x**2 + time, y**2 + x, z**2 + y, x + 2.0 * y + 3.0 * z))

    inputs = jnp.asarray([[0.2, 0.3, 0.4, 0.5], [0.7, -0.1, 0.2, 0.8]], dtype=jnp.float32)
    pressure_coefficient = 0.25
    viscosity_coefficient = 0.1
    groups = cartesian_3d_navier_stokes(
        predict_one,
        None,
        {"inputs": inputs},
        pressure_coefficient=pressure_coefficient,
        viscosity_coefficient=viscosity_coefficient,
    )
    residuals = jnp.concatenate([group[0] for group in groups], axis=-1)
    x, y, z, time = np.asarray(inputs).T
    u_x = x**2 + time
    u_y = y**2 + x
    u_z = z**2 + y
    expected = np.stack(
        (
            2.0 * (x + y + z),
            1.0 + 2.0 * x * u_x + pressure_coefficient - 2.0 * viscosity_coefficient,
            u_x + 2.0 * y * u_y + 2.0 * pressure_coefficient - 2.0 * viscosity_coefficient,
            u_y + 2.0 * z * u_z + 3.0 * pressure_coefficient - 2.0 * viscosity_coefficient,
        ),
        axis=-1,
    )

    assert residuals.dtype == jnp.float32
    np.testing.assert_allclose(np.asarray(residuals), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("equation", "input_size", "group_count"),
    [(cartesian_2d_navier_stokes, 3, 3), (cartesian_3d_navier_stokes, 4, 4)],
)
def test_cartesian_equations_return_named_singleton_groups(
    equation: Callable[..., Any],
    input_size: int,
    group_count: int,
) -> None:
    """Verify both framework-facing equations expose one group per declared residual.

    Args:
        equation: Configurable Cartesian residual equation.
        input_size: Required coordinate width and model-output width.
        group_count: Expected number of named residual groups.
    """

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Return a connected constant vector with the required output width.

        Args:
            params: Unused explicit-state placeholder.
            point: Cartesian coordinate vector.

        Returns:
            Zero velocity and pressure values connected to `point`.
        """
        del params
        return jnp.zeros((input_size,), dtype=point.dtype) + point[0] * 0.0

    batch = {"inputs": jnp.ones((1, input_size), dtype=jnp.float32)}
    groups = equation(predict_one, None, batch)

    assert len(groups) == group_count
    assert all(len(group) == 1 and group[0].shape == (1, 1) for group in groups)


def test_cartesian_viscous_residuals_have_finite_parameter_gradients() -> None:
    """Verify parameter differentiation through three-dimensional velocity Laplacians remains finite."""

    def predict_one(params: jax.Array, point: jax.Array) -> jax.Array:
        """Evaluate parameterized quadratic velocity fields.

        Args:
            params: Three trainable velocity coefficients.
            point: Coordinate vector ordered as `[x, y, z, t]`.

        Returns:
            Output vector ordered as `[u_x, u_y, u_z, p]`.
        """
        x, y, z, time = point
        return jnp.stack((params[0] * x**2 + time, params[1] * y**2, params[2] * z**2, x + y + z))

    inputs = jnp.asarray([[0.2, 0.3, 0.4, 0.5], [0.7, -0.1, 0.2, 0.8]], dtype=jnp.float32)

    def residual_loss(params: jax.Array) -> jax.Array:
        """Reduce viscous residuals to a differentiable scalar.

        Args:
            params: Three trainable velocity coefficients.

        Returns:
            Mean squared Cartesian residual.
        """
        groups = cartesian_3d_navier_stokes(
            predict_one,
            params,
            {"inputs": inputs},
            viscosity_coefficient=0.2,
        )
        residuals = jnp.concatenate([group[0] for group in groups], axis=-1)
        return jnp.mean(residuals**2)

    gradients = jax.grad(residual_loss)(jnp.asarray([1.0, 0.5, -0.25], dtype=jnp.float32))
    assert gradients.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(gradients)))


@pytest.mark.parametrize(
    ("equation", "width", "coordinate_label"),
    [
        (cartesian_2d_navier_stokes, 3, "Cartesian 2D"),
        (cartesian_3d_navier_stokes, 4, "Cartesian 3D"),
    ],
)
def test_cartesian_residuals_validate_shape_and_viscosity(
    equation: Callable[..., Any],
    width: int,
    coordinate_label: str,
) -> None:
    """Verify Cartesian equations reject invalid coordinate widths and viscosity.

    Args:
        equation: Cartesian residual equation under test.
        width: Valid coordinate and output width.
        coordinate_label: Coordinate-system label expected in the error.
    """

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Return a connected zero output vector.

        Args:
            params: Unused explicit-state placeholder.
            point: Cartesian coordinate vector.

        Returns:
            Zero model output with the valid width.
        """
        del params
        return jnp.zeros((width,), dtype=point.dtype) + point[0] * 0.0

    with pytest.raises(ValueError, match=coordinate_label):
        equation(predict_one, None, {"inputs": jnp.ones((2, width + 1))})
    with pytest.raises(ValueError, match="viscosity_coefficient"):
        equation(predict_one, None, {"inputs": jnp.ones((2, width))}, viscosity_coefficient=-0.1)


@pytest.mark.parametrize("equation", [cartesian_2d_navier_stokes, cartesian_3d_navier_stokes])
def test_cartesian_equations_reject_output_stream(equation: Callable[..., Any]) -> None:
    """Verify Cartesian Navier--Stokes equations expose only residual streams.

    Args:
        equation: Framework-facing Cartesian equation under test.
    """
    width = 3 if equation is cartesian_2d_navier_stokes else 4

    def predict_one(params: None, point: jax.Array) -> jax.Array:
        """Return a connected zero output vector.

        Args:
            params: Unused explicit-state placeholder.
            point: Cartesian coordinate vector.

        Returns:
            Zero model output matching the coordinate width.
        """
        del params
        return jnp.zeros((width,), dtype=point.dtype) + point[0] * 0.0

    with pytest.raises(ValueError, match="does not expose"):
        equation(predict_one, None, {"inputs": jnp.ones((1, width))}, stream="output")


def test_cartesian_equations_can_be_jitted() -> None:
    """Verify public Cartesian equations remain compatible with ordinary JAX compilation."""

    def predict_one(params: jax.Array, point: jax.Array) -> jax.Array:
        """Evaluate a linear two-dimensional field.

        Args:
            params: Scalar coefficient shared by the velocity components.
            point: Coordinate vector ordered as `[x, y, t]`.

        Returns:
            Output vector ordered as `[u_x, u_y, p]`.
        """
        x, y, time = point
        return jnp.stack((params * x + time, params * y, x + y))

    compiled = jax.jit(partial(cartesian_2d_navier_stokes, predict_one))
    groups = compiled(jnp.asarray(0.5), {"inputs": jnp.ones((2, 3), dtype=jnp.float32)})
    residuals = jnp.concatenate([group[0] for group in groups], axis=-1)
    assert residuals.shape == (2, 3)
    assert bool(jnp.all(jnp.isfinite(residuals)))
