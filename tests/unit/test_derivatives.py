import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.derivatives import hessian_diagonal, value_and_jacobian


def _vector_function(scale: jax.Array, x: jax.Array, y: jax.Array) -> jax.Array:
    """Evaluate a two-output polynomial with analytic derivatives.

    Args:
        scale: Scalar differentiable model parameter.
        x: First scalar coordinate.
        y: Second scalar coordinate.

    Returns:
        Polynomial output vector.
    """
    return jnp.stack((scale * x**2 + y, scale * x * y + y**3))


def _sum_values(values: jax.Array) -> jax.Array:
    """Reduce an array for scalar-argument validation.

    Args:
        values: Candidate differentiated array.

    Returns:
        Scalar array sum.
    """
    return jnp.sum(values)


def test_value_and_jacobian_matches_analytic_selected_derivatives() -> None:
    """Verify primal values and ordered selected derivatives under JIT compilation."""
    differentiated = jax.jit(value_and_jacobian(_vector_function, (1, 2)))
    scale = jnp.asarray(2.0, dtype=jnp.float32)
    x = jnp.asarray(3.0, dtype=jnp.float32)
    y = jnp.asarray(4.0, dtype=jnp.float32)

    value, (derivative_x, derivative_y) = differentiated(scale, x, y)

    np.testing.assert_allclose(value, [22.0, 88.0])
    np.testing.assert_allclose(derivative_x, [12.0, 8.0])
    np.testing.assert_allclose(derivative_y, [1.0, 54.0])


def test_hessian_diagonal_computes_only_pure_second_derivatives() -> None:
    """Verify ordered Hessian-diagonal entries against an analytic vector function."""
    differentiated = jax.jit(hessian_diagonal(_vector_function, (1, 2)))
    scale = jnp.asarray(2.0, dtype=jnp.float32)
    x = jnp.asarray(3.0, dtype=jnp.float32)
    y = jnp.asarray(4.0, dtype=jnp.float32)

    second_x, second_y = differentiated(scale, x, y)

    np.testing.assert_allclose(second_x, [4.0, 0.0])
    np.testing.assert_allclose(second_y, [0.0, 24.0])


def test_selective_derivatives_preserve_finite_parameter_gradients() -> None:
    """Verify forward coordinate derivatives remain differentiable with respect to model parameters."""
    value_and_first = value_and_jacobian(_vector_function, (1, 2))
    pure_second = hessian_diagonal(_vector_function, 1)

    def derivative_loss(scale: jax.Array) -> jax.Array:
        """Reduce values and coordinate derivatives to a scalar parameter loss.

        Args:
            scale: Scalar differentiable model parameter.

        Returns:
            Sum of primal, first-derivative, and pure-second-derivative values.
        """
        value, first = value_and_first(scale, jnp.asarray(0.5), jnp.asarray(-0.25))
        second = pure_second(scale, jnp.asarray(0.5), jnp.asarray(-0.25))
        return jnp.sum(value) + sum(jnp.sum(entry) for entry in (*first, *second))

    gradient = jax.grad(derivative_loss)(jnp.asarray(1.5, dtype=jnp.float32))

    assert gradient.shape == ()
    assert bool(jnp.isfinite(gradient))


@pytest.mark.parametrize(
    ("argnums", "error_type", "message"),
    [
        ((), ValueError, "non-empty and unique"),
        ((1, 1), ValueError, "non-empty and unique"),
        ((1.5,), TypeError, "integer"),
        (True, TypeError, "integer"),
    ],
)
def test_selective_derivatives_validate_argument_numbers(
    argnums: object,
    error_type: type[Exception],
    message: str,
) -> None:
    """Verify invalid derivative argument selections fail before tracing.

    Args:
        argnums: Invalid positional argument selection.
        error_type: Expected validation exception type.
        message: Expected exception-message fragment.
    """
    with pytest.raises(error_type, match=message):
        value_and_jacobian(_vector_function, argnums)  # type: ignore[arg-type]


def test_selective_derivatives_reject_nonscalar_selected_arguments() -> None:
    """Verify vector primals cannot silently become directional rather than complete derivatives."""
    differentiated = value_and_jacobian(_sum_values, 0)

    with pytest.raises(ValueError, match="scalar"):
        differentiated(jnp.ones(2, dtype=jnp.float32))


def test_selective_derivatives_pass_gradients_to_a_closed_over_model_state() -> None:
    """Verify derivatives of `u(x, y)` that closes over the model state, as equations write it, keep state gradients."""

    def model_apply(scale: jax.Array, point: jax.Array) -> jax.Array:
        """Apply the polynomial model to one stacked point.

        Args:
            scale: Scalar differentiable model parameter.
            point: Coordinates `[x, y]`.

        Returns:
            Polynomial output vector.
        """
        return _vector_function(scale, point[0], point[1])

    x = jnp.asarray(3.0, dtype=jnp.float32)
    y = jnp.asarray(4.0, dtype=jnp.float32)

    def derivatives(scale: jax.Array) -> jax.Array:
        """Evaluate output 1 and its selected derivatives with the state closed over.

        Args:
            scale: Scalar differentiable model parameter.

        Returns:
            Stacked `[u, du_dx, du_dy, d2u_dy2]`.
        """

        def u(x_coord: jax.Array, y_coord: jax.Array) -> jax.Array:
            """Evaluate model output 1 at one point.

            Args:
                x_coord: Scalar first coordinate.
                y_coord: Scalar second coordinate.

            Returns:
                Scalar output `scale * x * y + y**3`.
            """
            return model_apply(scale, jnp.stack((x_coord, y_coord)))[1]

        value, (du_dx, du_dy) = value_and_jacobian(u, (0, 1))(x, y)
        (d2u_dy2,) = hessian_diagonal(u, 1)(x, y)
        return jnp.stack((value, du_dx, du_dy, d2u_dy2))

    scale = jnp.asarray(2.0, dtype=jnp.float32)
    np.testing.assert_allclose(derivatives(scale), [88.0, 8.0, 54.0, 24.0])
    # u = scale * x * y + y**3, du_dx = scale * y, and du_dy = scale * x + 3 * y**2 depend on `scale`, so the gradient
    # of their sum is x * y + y + x.
    np.testing.assert_allclose(jax.grad(lambda state: jnp.sum(derivatives(state)))(scale), 19.0)
