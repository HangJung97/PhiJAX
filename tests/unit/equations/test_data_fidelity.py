from functools import partial

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
    with pytest.raises(ValueError, match="one projected target"):
        phase_wrapped_fidelity(
            lambda state, point: point,
            None,
            batch,
            output_indices=(0, 1),
            target_indices=(0,),
        )


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("projection_key", ["projection", "direction"])
@pytest.mark.parametrize("sample_count", [1, 2])
def test_fidelity_projection_keys(wrapped: bool, projection_key: str, sample_count: int) -> None:
    """Verify named projections, selected component order, and compiled gradients.

    Args:
        wrapped: Whether to evaluate periodic fidelity.
        projection_key: Batch field containing projection directions.
        sample_count: Number of observations, including the singleton case.
    """
    equation = phase_wrapped_fidelity if wrapped else base_data_fidelity
    inputs = jnp.asarray([[0.25, 9.0, 0.5], [0.75, 8.0, 0.25]], dtype=jnp.float32)[:sample_count]
    batch = {
        "inputs": inputs,
        "targets": jnp.asarray([[9.0, -0.25], [8.0, -0.5]], dtype=jnp.float32)[:sample_count],
        "period": jnp.asarray([[1.0], [2.0]], dtype=jnp.float32)[:sample_count],
        "weight": jnp.asarray([[2.0], [0.0]], dtype=jnp.float32)[:sample_count],
        "projection": jnp.zeros((sample_count, 2), dtype=jnp.float32),
        projection_key: jnp.asarray([[0.5, 1.0], [1.0, -1.0]], dtype=jnp.float32)[:sample_count],
    }

    def evaluate(scale: jax.Array, points: jax.Array, stream: str = "residual") -> tuple:
        """Evaluate the configured fidelity with explicit model state and inputs.

        Args:
            scale: Trainable scalar multiplier.
            points: Model input coordinates.
            stream: Residual or unprojected output stream.

        Returns:
            Equation groups for the requested stream.
        """
        return equation(
            lambda state, point: state * point,
            scale,
            {**batch, "inputs": points},
            output_indices=(2, 0),
            target_indices=(1,),
            projection_key=projection_key,
            target_negation=True,
            stream=stream,
        )

    groups = jax.jit(evaluate)(jnp.asarray(1.0), inputs)
    projected = np.asarray([[0.5], [-0.5]], dtype=np.float32)[:sample_count]
    target = -np.asarray(batch["targets"][:, 1:2])
    weight = np.asarray(batch["weight"])
    if wrapped:
        period = np.asarray(batch["period"])
        expected = (
            weight * (np.cos(np.pi * projected / period) - np.cos(np.pi * target / period)),
            weight * (np.sin(np.pi * projected / period) - np.sin(np.pi * target / period)),
        )
    else:
        expected = (weight * (projected - target),)
    for actual, reference in zip(groups[0], expected, strict=True):
        np.testing.assert_allclose(actual, reference, atol=1e-6)
        assert actual.shape == (sample_count, 1)
        assert actual.dtype == jnp.float32
    np.testing.assert_allclose(evaluate(jnp.asarray(1.0), inputs, "output")[0][0], inputs[:, (2, 0)])
    gradients = jax.jit(jax.grad(lambda scale, points: sum(jnp.sum(r**2) for r in evaluate(scale, points)[0]), (0, 1)))(
        jnp.asarray(1.0), inputs
    )
    for gradient in gradients:
        assert gradient.dtype == jnp.float32
        assert bool(jnp.all(jnp.isfinite(gradient)))
    assert bool(jnp.any(gradients[1] != 0))


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("projection_key", [None, "absent"])
def test_fidelity_can_skip_projection(wrapped: bool, projection_key: str | None) -> None:
    """Verify disabled or missing projection fields retain direct supervision.

    Args:
        wrapped: Whether to evaluate periodic fidelity.
        projection_key: Disabled or absent projection field.
    """
    equation = phase_wrapped_fidelity if wrapped else base_data_fidelity
    batch = {
        "inputs": jnp.asarray([[0.25, 0.5]], dtype=jnp.float32),
        "targets": jnp.asarray([[0.25, 0.5]], dtype=jnp.float32),
        "period": jnp.ones((1, 1), dtype=jnp.float32),
        "projection": jnp.zeros((1, 2), dtype=jnp.float32),
    }
    groups = equation(
        lambda state, point: point,
        None,
        batch,
        output_indices=(0, 1),
        target_indices=(0, 1),
        projection_key=projection_key,
    )
    for residual in groups[0]:
        np.testing.assert_array_equal(residual, jnp.zeros((1, 2)))


@pytest.mark.parametrize("wrapped", [False, True])
def test_fidelity_projection_requires_scalar_target(wrapped: bool) -> None:
    """Verify projected supervision rejects multiple target components.

    Args:
        wrapped: Whether to evaluate periodic fidelity.
    """
    equation = phase_wrapped_fidelity if wrapped else base_data_fidelity
    with pytest.raises(ValueError, match="one projected target"):
        equation(
            lambda state, point: point,
            None,
            {"direction": jnp.ones((1, 2))},
            output_indices=(0, 1),
            target_indices=(0, 1),
            projection_key="direction",
        )


def test_phase_wrapped_residuals_reject_projection_shape_mismatch() -> None:
    """Verify wrapped projection directions cannot broadcast across output components."""
    with pytest.raises(ValueError, match="matching shapes"):
        phase_wrapped_residuals(jnp.ones((2, 3)), jnp.ones((2, 1)), jnp.ones((2, 1)), projection=jnp.ones((2, 2)))


@pytest.mark.parametrize("projected", [False, True])
@pytest.mark.parametrize("include_default_period", [False, True])
def test_phase_wrapped_fidelity_custom_period_key(projected: bool, include_default_period: bool) -> None:
    """Verify custom periods control compiled residuals with optional projection.

    Args:
        projected: Whether to project vector predictions before phase conversion.
        include_default_period: Whether an unused default period field is present.
    """
    batch = {
        "inputs": jnp.asarray([[0.25, 0.5], [0.5, 0.25]], dtype=jnp.float32),
        "targets": jnp.asarray([[0.5], [0.25]], dtype=jnp.float32),
        "wrapping_period": jnp.asarray([[0.5], [2.0]], dtype=jnp.float32),
    }
    if projected:
        batch["direction"] = jnp.asarray([[1.0, 0.5], [0.5, 1.0]], dtype=jnp.float32)
    if include_default_period:
        batch["period"] = jnp.full((2, 1), 3.0, dtype=jnp.float32)
    evaluate = jax.jit(
        partial(
            phase_wrapped_fidelity,
            lambda state, point: point,
            None,
            output_indices=(0, 1) if projected else (0,),
            projection_key="direction",
            period_key="wrapping_period",
        )
    )

    ((cosine, sine),) = evaluate(batch)

    output = np.full((2, 1), 0.5, dtype=np.float32) if projected else np.asarray(batch["inputs"][:, :1])
    output_phase = np.pi * output / np.asarray(batch["wrapping_period"])
    target_phase = np.pi * np.asarray(batch["targets"]) / np.asarray(batch["wrapping_period"])
    np.testing.assert_allclose(cosine, np.cos(output_phase) - np.cos(target_phase), atol=1e-6)
    np.testing.assert_allclose(sine, np.sin(output_phase) - np.sin(target_phase), atol=1e-6)
    assert cosine.shape == sine.shape == (2, 1)
    assert cosine.dtype == sine.dtype == jnp.float32


def test_phase_wrapped_fidelity_missing_custom_period_key() -> None:
    """Require the configured period for residuals while allowing the output stream."""
    batch = {
        "inputs": jnp.asarray([[0.25]], dtype=jnp.float32),
        "targets": jnp.zeros((1, 1), dtype=jnp.float32),
        "period": jnp.ones((1, 1), dtype=jnp.float32),
    }
    evaluate = partial(phase_wrapped_fidelity, lambda state, point: point, None, period_key="wrapping_period")
    with pytest.raises(KeyError, match="wrapping_period"):
        evaluate(batch)
    np.testing.assert_array_equal(evaluate(batch, stream="output")[0][0], batch["inputs"])


@pytest.mark.parametrize("wrapped", [False, True])
@pytest.mark.parametrize("weight_key", ["weight", "confidence", None, "absent"])
def test_fidelity_weight_keys(wrapped: bool, weight_key: str | None) -> None:
    """Verify named weights, disabled weighting, and unweighted output streams under JIT.

    Args:
        wrapped: Whether to evaluate periodic fidelity.
        weight_key: Default, custom, disabled, or absent weight field.
    """
    equation = phase_wrapped_fidelity if wrapped else base_data_fidelity
    batch = {
        "inputs": jnp.asarray([[0.25, 0.5], [0.5, 0.25]], dtype=jnp.float32),
        "targets": jnp.zeros((2, 1), dtype=jnp.float32),
        "direction": jnp.asarray([[1.0, 0.5], [0.5, 1.0]], dtype=jnp.float32),
        "period": jnp.ones((2, 1), dtype=jnp.float32),
        "weight": jnp.asarray([[3.0], [4.0]], dtype=jnp.float32),
        "confidence": jnp.asarray([[2.0], [0.0]], dtype=jnp.float32),
    }
    evaluate = partial(
        equation,
        lambda state, point: point,
        None,
        output_indices=(0, 1),
        projection_key="direction",
        weight_key=weight_key,
    )
    groups = jax.jit(evaluate)(batch)
    if weight_key in ("weight", "confidence"):
        weight = np.asarray(batch[weight_key])
    else:
        weight = np.ones((2, 1), dtype=np.float32)
    expected = (-weight, weight) if wrapped else (0.5 * weight,)
    for actual, reference in zip(groups[0], expected, strict=True):
        np.testing.assert_allclose(actual, reference, atol=1e-6)
        assert actual.shape == (2, 1)
        assert actual.dtype == jnp.float32
    output = jax.jit(partial(evaluate, stream="output"))(batch)
    np.testing.assert_array_equal(output[0][0], batch["inputs"])
