from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.equations import base_boundary_residual, free_slip_boundary, free_slip_residual, no_slip_residual


def test_base_boundary_returns_direct_component_residuals() -> None:
    """Verify the base boundary residual compares every configured component."""
    output = jnp.asarray([[2.0, 3.0], [4.0, 5.0]], dtype=jnp.float32)
    target = jnp.asarray([[1.0, 1.0], [1.0, 2.0]], dtype=jnp.float32)

    residual = base_boundary_residual(output, target)

    np.testing.assert_allclose(residual, [[1.0, 2.0], [3.0, 3.0]])


def test_no_slip_constrains_all_velocity_components_with_finite_gradients() -> None:
    """Verify no-slip retains vector residuals and differentiates with respect to velocity."""
    velocity = jnp.asarray([[2.0, -1.0]], dtype=jnp.float32)
    target = jnp.zeros_like(velocity)

    residual = no_slip_residual(velocity, target)
    gradient = jax.grad(lambda values: jnp.sum(no_slip_residual(values, target) ** 2))(velocity)

    np.testing.assert_allclose(residual, velocity)
    np.testing.assert_allclose(gradient, 2.0 * velocity)


def test_base_boundary_rejects_mismatched_component_widths() -> None:
    """Verify boundary component mismatches fail before subtraction."""
    with pytest.raises(ValueError, match="widths must match"):
        base_boundary_residual(jnp.ones((2, 2)), jnp.ones((2, 1)))


def test_free_slip_projects_velocity_error_onto_normal() -> None:
    """Verify that only the wall-normal velocity mismatch contributes to a free-slip residual."""
    velocity = jnp.asarray([[2.0, 3.0], [4.0, 5.0]])
    target = jnp.asarray([[1.0, 1.0], [1.0, 2.0]])
    normals = jnp.asarray([[1.0, 0.0], [0.0, -1.0]])
    np.testing.assert_allclose(free_slip_residual(velocity, target, normals), [[1.0], [-3.0]])


def test_free_slip_boundary_exposes_residual_and_output_groups() -> None:
    """Verify the free-slip equation supports residual losses and output-based NTK diagnostics."""

    def predict_one(state: None, point: jax.Array) -> jax.Array:
        """Return the input as a deterministic velocity prediction.

        Args:
            state: Unused model-state placeholder.
            point: Two-component input and velocity vector.

        Returns:
            Unchanged two-component prediction.
        """
        del state
        return point

    batch = {
        "inputs": jnp.asarray([[2.0, 3.0], [4.0, 5.0]]),
        "targets": jnp.asarray([[1.0, 1.0], [1.0, 2.0]]),
        "normals": jnp.asarray([[1.0, 0.0], [0.0, -1.0]]),
    }
    residual_groups = free_slip_boundary(predict_one, None, batch)
    output_groups = free_slip_boundary(predict_one, None, batch, stream="output")

    np.testing.assert_allclose(residual_groups[0][0], [[1.0], [-3.0]])
    np.testing.assert_allclose(output_groups[0][0], batch["inputs"])


def test_free_slip_boundary_rejects_invalid_component_selection() -> None:
    """Verify duplicated output components fail before boundary residual evaluation."""
    batch = {
        "inputs": jnp.ones((2, 2)),
        "targets": jnp.ones((2, 2)),
        "normals": jnp.ones((2, 2)),
    }
    with pytest.raises(ValueError, match="unique nonnegative"):
        free_slip_boundary(lambda state, point: point, None, batch, output_indices=(0, 0))


@pytest.mark.parametrize("include_default_normals", [False, True])
def test_free_slip_boundary_custom_normals_key(include_default_normals: bool) -> None:
    """Verify custom normals project selected components under JIT.

    Args:
        include_default_normals: Whether an unused default normals field is present.
    """
    batch = {
        "inputs": jnp.asarray([[2.0, 9.0, 3.0], [4.0, 8.0, 5.0]], dtype=jnp.float32),
        "targets": jnp.asarray([[1.0, 1.0], [1.0, 2.0]], dtype=jnp.float32),
        "wall_normals": jnp.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=jnp.float32),
    }
    if include_default_normals:
        batch["normals"] = jnp.zeros((2, 2), dtype=jnp.float32)
    evaluate = partial(
        free_slip_boundary,
        lambda state, point: point,
        None,
        output_indices=(2, 0),
        target_indices=(1, 0),
        normals_key="wall_normals",
    )
    ((residual,),) = jax.jit(evaluate)(batch)
    np.testing.assert_allclose(residual, [[2.0], [-3.0]])
    assert residual.shape == (2, 1)
    assert residual.dtype == jnp.float32
    output = jax.jit(partial(evaluate, stream="output"))(batch)
    np.testing.assert_array_equal(output[0][0], batch["inputs"][:, (2, 0)])


def test_free_slip_boundary_missing_custom_normals_key() -> None:
    """Require configured normals for residuals while allowing the output stream."""
    batch = {
        "inputs": jnp.asarray([[2.0, 3.0]], dtype=jnp.float32),
        "targets": jnp.zeros((1, 2), dtype=jnp.float32),
        "normals": jnp.ones((1, 2), dtype=jnp.float32),
    }
    evaluate = partial(free_slip_boundary, lambda state, point: point, None, normals_key="wall_normals")
    with pytest.raises(KeyError, match="wall_normals"):
        evaluate(batch)
    np.testing.assert_array_equal(evaluate(batch, stream="output")[0][0], batch["inputs"])
