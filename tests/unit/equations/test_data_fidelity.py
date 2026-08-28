import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.equations import (
    base_data_fidelity,
    base_data_fidelity_residual,
    phase_wrapped_fidelity,
    phase_wrapped_residuals,
)


def test_base_data_fidelity_supports_weight_and_target_negation() -> None:
    """Verify unwrapped residuals apply target sign and weights before reduction."""
    output = jnp.asarray([[2.0], [4.0]], dtype=jnp.float32)
    target = jnp.asarray([[1.0], [1.0]], dtype=jnp.float32)
    weight = jnp.asarray([[0.5], [2.0]], dtype=jnp.float32)

    residual = base_data_fidelity_residual(output, target, weight=weight, target_negation=True)

    np.testing.assert_allclose(residual, [[1.5], [10.0]])


def test_base_data_fidelity_projects_vector_outputs_before_comparison() -> None:
    """Verify a generic vector projection produces scalar supervised residuals."""
    output = jnp.asarray([[3.0, 4.0], [5.0, 12.0]], dtype=jnp.float32)
    projection = jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)
    target = jnp.asarray([[4.0], [3.0]], dtype=jnp.float32)

    residual = base_data_fidelity_residual(output, target, projection=projection)
    gradient = jax.grad(
        lambda prediction: jnp.sum(base_data_fidelity_residual(prediction, target, projection=projection) ** 2)
    )(output)

    np.testing.assert_allclose(residual, [[0.0], [2.0]])
    assert gradient.shape == output.shape
    assert bool(jnp.all(jnp.isfinite(gradient)))


def test_base_data_fidelity_rejects_projection_shape_mismatch() -> None:
    """Verify incompatible projection directions fail before broadcasting."""
    with pytest.raises(ValueError, match="matching shapes"):
        base_data_fidelity_residual(jnp.ones((2, 3)), jnp.ones((2, 1)), projection=jnp.ones((2, 2)))


def test_base_data_fidelity_exposes_direct_residual_and_output_groups() -> None:
    """Verify direct supervised batches compose with generic residual terms and output-based balancing."""

    def predict_one(scale: jax.Array, point: jax.Array) -> jax.Array:
        """Scale a two-component point.

        Args:
            scale: Trainable scalar multiplier.
            point: Two-component model input.

        Returns:
            Scaled two-component prediction.
        """
        return scale * point

    batch = {
        "inputs": jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32),
        "targets": jnp.asarray([[0.5], [1.5]], dtype=jnp.float32),
        "weight": jnp.asarray([[2.0], [0.5]], dtype=jnp.float32),
    }
    residual_groups = base_data_fidelity(
        predict_one,
        jnp.asarray(1.0),
        batch,
        output_indices=(0,),
        target_indices=(0,),
    )
    output_groups = base_data_fidelity(
        predict_one,
        jnp.asarray(1.0),
        batch,
        output_indices=(0,),
        target_indices=(0,),
        stream="output",
    )

    np.testing.assert_allclose(residual_groups[0][0], [[1.0], [0.75]])
    np.testing.assert_allclose(output_groups[0][0], [[1.0], [3.0]])


def test_base_data_fidelity_projects_selected_vector_outputs() -> None:
    """Verify configured vector outputs can be projected against one scalar target component."""
    batch = {
        "inputs": jnp.asarray([[3.0, 4.0], [5.0, 12.0]], dtype=jnp.float32),
        "targets": jnp.asarray([[4.0], [3.0]], dtype=jnp.float32),
        "projection": jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32),
    }
    groups = base_data_fidelity(
        lambda state, point: point,
        None,
        batch,
        output_indices=(0, 1),
        target_indices=(0,),
    )
    np.testing.assert_allclose(groups[0][0], [[0.0], [2.0]])


def test_phase_wrapped_residuals_include_target_negation_and_weight() -> None:
    """Verify target negation and residual weighting in phase-wrapped fidelity."""
    output = jnp.asarray([[0.25]], dtype=jnp.float32)
    target = jnp.asarray([[-0.25]], dtype=jnp.float32)
    cosine, sine = phase_wrapped_residuals(
        output,
        target,
        jnp.ones_like(output),
        weight=jnp.asarray([[2.0]], dtype=jnp.float32),
        target_negation=True,
    )
    np.testing.assert_allclose(cosine, 0.0, atol=1e-7)
    np.testing.assert_allclose(sine, 0.0, atol=1e-7)


def test_phase_wrapped_fidelity_exposes_residual_and_output_groups() -> None:
    """Verify phase fidelity groups residual losses and exposes output-based NTK diagnostics."""

    def predict_one(state: None, point: jax.Array) -> jax.Array:
        """Return two deterministic output components.

        Args:
            state: Unused model-state placeholder.
            point: Two-component input point.

        Returns:
            Two-component prediction.
        """
        del state
        return 2.0 * point

    batch = {
        "inputs": jnp.asarray([[0.25, 1.0], [0.5, 2.0]], dtype=jnp.float32),
        "targets": jnp.zeros((2, 1), dtype=jnp.float32),
        "period": jnp.ones((2, 1), dtype=jnp.float32),
    }
    residual_groups = phase_wrapped_fidelity(
        predict_one,
        None,
        batch,
        output_indices=(0,),
        target_indices=(0,),
    )
    output_groups = phase_wrapped_fidelity(
        predict_one,
        None,
        batch,
        output_indices=(0,),
        target_indices=(0,),
        stream="output",
    )

    assert len(residual_groups) == 1
    assert len(residual_groups[0]) == 2
    np.testing.assert_allclose(output_groups[0][0], [[0.5], [1.0]])


def test_phase_wrapped_fidelity_validates_static_component_policy() -> None:
    """Verify incompatible component selections fail when the configured equation is evaluated."""
    batch = {
        "inputs": jnp.ones((2, 2)),
        "targets": jnp.ones((2, 1)),
        "period": jnp.ones((2, 1)),
    }
    with pytest.raises(ValueError, match="equal lengths"):
        phase_wrapped_fidelity(
            lambda state, point: point,
            None,
            batch,
            output_indices=(0, 1),
            target_indices=(0,),
        )
