import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.equations.pde import burgers_1d


def _polynomial_solution(parameters: jax.Array, point: jax.Array) -> jax.Array:
    """Evaluate a scalar manufactured Burgers field.

    Args:
        parameters: Coefficients `[a, b]` defining `u = a*t + b*x**2`.
        point: Coordinate vector ordered as `[t, x]`.

    Returns:
        Singleton model-output vector.
    """
    time, position = point
    return jnp.asarray([parameters[0] * time + parameters[1] * position**2])


def test_burgers_residuals_match_manufactured_derivatives() -> None:
    """Verify convection, time derivative, and diffusion against an analytic polynomial field."""
    parameters = jnp.asarray([1.5, 0.75], dtype=jnp.float32)
    inputs = jnp.asarray([[0.2, -0.5], [0.7, 0.25]], dtype=jnp.float32)
    viscosity = 0.01 / math.pi

    residuals = burgers_1d(
        _polynomial_solution,
        parameters,
        {"inputs": inputs},
        viscosity_coefficient=viscosity,
    )[0][0]

    time, position = np.asarray(inputs).T
    a, b = np.asarray(parameters)
    values = a * time + b * position**2
    expected = a + values * (2.0 * b * position) - viscosity * (2.0 * b)
    np.testing.assert_allclose(np.asarray(residuals[:, 0]), expected, rtol=1e-6, atol=1e-6)


def test_burgers_grouped_residual_has_finite_parameter_gradients() -> None:
    """Verify the framework-facing residual retains shape and finite gradients through `u_xx`."""
    batch = {"inputs": jnp.asarray([[0.2, -0.5], [0.7, 0.25]], dtype=jnp.float32)}
    groups = burgers_1d(_polynomial_solution, jnp.asarray([1.5, 0.75]), batch)

    def loss(parameters: jax.Array) -> jax.Array:
        """Reduce a grouped Burgers residual to a scalar.

        Args:
            parameters: Manufactured solution coefficients.

        Returns:
            Sum of squared residual values.
        """
        residual = burgers_1d(_polynomial_solution, parameters, batch)[0][0]
        return jnp.sum(residual**2)

    gradients = jax.grad(loss)(jnp.asarray([1.5, 0.75], dtype=jnp.float32))
    assert len(groups) == 1
    assert groups[0][0].shape == (2, 1)
    assert gradients.shape == (2,)
    assert bool(jnp.all(jnp.isfinite(gradients)))


def test_inviscid_burgers_path_omits_diffusion() -> None:
    """Verify zero viscosity produces the analytic first-order residual."""
    parameters = jnp.asarray([1.5, 0.75], dtype=jnp.float32)
    inputs = jnp.asarray([[0.2, -0.5]], dtype=jnp.float32)
    residual = burgers_1d(
        _polynomial_solution,
        parameters,
        {"inputs": inputs},
        viscosity_coefficient=0.0,
    )[0][0]
    value = 1.5 * 0.2 + 0.75 * 0.5**2
    np.testing.assert_allclose(residual, [[1.5 + value * 2.0 * 0.75 * -0.5]])


@pytest.mark.parametrize(
    ("inputs", "kwargs", "match"),
    [
        (jnp.ones((2, 3)), {}, "columns"),
        (jnp.ones((2, 2)), {"viscosity_coefficient": -1.0}, "nonnegative"),
        (jnp.ones((2, 2)), {"output_index": -1}, "output_index"),
    ],
)
def test_burgers_residuals_reject_invalid_settings(
    inputs: jax.Array,
    kwargs: dict[str, float | int],
    match: str,
) -> None:
    """Verify invalid coordinate and equation policies fail before differentiation.

    Args:
        inputs: Invalid or valid candidate coordinate array.
        kwargs: Invalid equation keyword arguments.
        match: Expected validation-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        burgers_1d(_polynomial_solution, jnp.ones(2), {"inputs": inputs}, **kwargs)


def test_burgers_rejects_raw_output_streams() -> None:
    """Verify PDE terms expose residual sensitivities rather than raw network outputs."""
    batch = {"inputs": jnp.ones((2, 2), dtype=jnp.float32)}
    with pytest.raises(ValueError, match="does not expose"):
        burgers_1d(_polynomial_solution, jnp.ones(2), batch, stream="output")
