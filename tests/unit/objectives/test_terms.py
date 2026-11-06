from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import pytest

from phijax.equations import free_slip_boundary, phase_wrapped_fidelity, polar_navier_stokes, residual_equation
from phijax.objectives import CompositeObjective, ResidualTerm
from phijax.types import ArrayMapping, ModelApply, ResidualGroups, ResidualStream

LOSS_NAMES = (
    "fidelity/uR",
    "boundary/free_slip",
    "pde/continuity",
    "pde/momentum_r",
    "pde/momentum_th",
)


def _objective(*, viscosity_coefficient: float = 0.0) -> CompositeObjective:
    """Build a generic multi-term composition fixture.

    Args:
        viscosity_coefficient: Polar PDE viscosity coefficient.

    Returns:
        Generic composite objective containing three configured residual functions.
    """
    return CompositeObjective(
        {
            "fidelity": ResidualTerm(
                partial(
                    phase_wrapped_fidelity,
                    output_indices=(0,),
                    target_indices=(0,),
                    target_negation=True,
                ),
                batch_key="fidelity",
                names=(LOSS_NAMES[0],),
                ntk_stream="output",
            ),
            "boundary": ResidualTerm(
                residual_fn=partial(free_slip_boundary, output_indices=(0, 1), target_indices=(0, 1)),
                batch_key="boundary",
                ntk_stream="output",
            ),
            "pde": ResidualTerm(
                residual_fn=partial(polar_navier_stokes, viscosity_coefficient=viscosity_coefficient),
                batch_key="pde",
            ),
        }
    )


def _batches() -> dict[str, dict[str, jax.Array]]:
    """Build fixed-shape positive-radius batches for term tests.

    Returns:
        Named fidelity, boundary, and PDE batches with two samples per subset.
    """
    inputs = jnp.asarray([[0.5, 0.1, 0.0], [0.8, -0.2, 0.5]], dtype=jnp.float32)
    return {
        "fidelity": {
            "inputs": inputs,
            "targets": jnp.zeros((2, 1), dtype=jnp.float32),
            "period": jnp.ones((2, 1), dtype=jnp.float32),
            "weight": jnp.ones((2, 1), dtype=jnp.float32),
        },
        "boundary": {
            "inputs": inputs,
            "targets": jnp.zeros((2, 2), dtype=jnp.float32),
            "normals": jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        },
        "pde": {"inputs": inputs},
    }


def test_residual_term_reduces_groups_and_dispatches_ntk_streams() -> None:
    """Verify grouped arrays share one scalar loss and an equation can expose a separate output stream."""

    @residual_equation(names=("grouped",), default_ntk_stream="output")
    def grouped_residuals(
        model_apply: ModelApply,
        model_state: Any,
        batch: ArrayMapping,
        *,
        stream: ResidualStream = "residual",
    ) -> ResidualGroups:
        """Return deterministic residual or output groups for generic-term testing.

        Args:
            model_apply: Unused model application callable.
            model_state: Unused model state.
            batch: Batch containing a reference array.
            stream: Requested residual representation.

        Returns:
            One two-array residual group or one single-array output group.
        """
        del model_apply, model_state
        reference = batch["values"]
        return ((3.0 * reference,),) if stream == "output" else ((reference, 2.0 * reference),)

    term = ResidualTerm(grouped_residuals, batch_key="data", names=("grouped",))
    residual_term = ResidualTerm(grouped_residuals, batch_key="data", names=("grouped",), ntk_stream="residual")
    batches = {"data": {"values": jnp.ones((2, 1), dtype=jnp.float32)}}
    losses = term.losses(lambda state, point: point, None, batches)
    stream = term.residual_stream("grouped", lambda state, point: point, None, batches)
    residual_stream = residual_term.residual_stream("grouped", lambda state, point: point, None, batches)

    assert float(losses["grouped"]) == pytest.approx(5.0)
    assert stream.shape == (2, 1)
    assert bool(jnp.all(stream == 3.0))
    assert residual_stream.shape == (2, 2)
    assert bool(jnp.all(residual_stream == jnp.asarray([[1.0, 2.0], [1.0, 2.0]])))


def test_composite_terms_expose_scalar_losses_and_raw_streams() -> None:
    """Verify configured functions preserve loss ordering and output-based NTK stream widths."""

    def analytic_model(state: None, point: jax.Array) -> jax.Array:
        """Evaluate a differentiable analytic polar field.

        Args:
            state: Unused explicit-state placeholder.
            point: Polar coordinate vector ordered as `[r, th, t]`.

        Returns:
            Analytic output vector ordered as `[u_r, u_th, p]`.
        """
        del state
        radius, theta, time = point
        return jnp.stack((radius + time, radius * theta, theta + time))

    objective = _objective()
    losses = objective.losses(analytic_model, None, _batches())
    streams = objective.residual_streams(analytic_model, None, _batches())
    assert tuple(losses) == LOSS_NAMES
    assert tuple(streams) == LOSS_NAMES
    assert all(value.shape == () for value in losses.values())
    assert streams["fidelity/uR"].shape == (2, 1)
    assert streams["boundary/free_slip"].shape == (2, 2)
    assert streams["pde/continuity"].shape == (2, 1)


def test_composite_objective_infers_terms_batch_keys_names_and_streams() -> None:
    """Verify equation shorthand preserves declaration order and decorated metadata."""
    objective = CompositeObjective.from_equations(
        {
            "boundary": partial(free_slip_boundary, output_indices=(0, 1), target_indices=(0, 1)),
            "pde": partial(polar_navier_stokes, viscosity_coefficient=0.0),
        }
    )

    assert objective.batch_keys == ("boundary", "pde")
    assert objective.loss_names == (
        "boundary/free_slip",
        "pde/continuity",
        "pde/momentum_r",
        "pde/momentum_th",
    )
    boundary_term, pde_term = objective.terms["boundary"], objective.terms["pde"]
    assert isinstance(boundary_term, ResidualTerm)
    assert isinstance(pde_term, ResidualTerm)
    assert boundary_term.ntk_stream == "output"
    assert pde_term.ntk_stream == "residual"


def test_polar_term_supports_nonzero_viscosity() -> None:
    """Verify a configured viscous PDE function evaluates finite losses and streams."""

    def analytic_model(state: None, point: jax.Array) -> jax.Array:
        """Evaluate a twice-differentiable manufactured polar field.

        Args:
            state: Unused explicit-state placeholder.
            point: Polar coordinate vector ordered as `[r, th, t]`.

        Returns:
            Polynomial output vector ordered as `[u_r, u_th, p]`.
        """
        del state
        radius, theta, time = point
        return jnp.stack((radius**2 + theta**2 + time, radius * theta + theta**2, radius + theta * time))

    objective = _objective(viscosity_coefficient=0.1)
    losses = objective.losses(analytic_model, None, _batches())
    momentum_stream = objective.residual_stream("pde/momentum_r", analytic_model, None, _batches())
    assert momentum_stream.shape == (2, 1)
    assert all(bool(jnp.isfinite(loss)) for loss in losses.values())


def test_residual_term_validates_configuration_and_equation_groups() -> None:
    """Verify invalid term settings and equation return structures fail clearly."""

    def one_group(
        model_apply: ModelApply,
        model_state: Any,
        batch: ArrayMapping,
        *,
        stream: ResidualStream = "residual",
    ) -> ResidualGroups:
        """Return one residual group regardless of the configured loss count.

        Args:
            model_apply: Unused model application callable.
            model_state: Unused model state.
            batch: Unused batch.
            stream: Unused requested stream.

        Returns:
            One single-array residual group.
        """
        del model_apply, model_state, batch, stream
        return ((jnp.ones((2, 1)),),)

    with pytest.raises(ValueError, match="unique non-empty"):
        ResidualTerm(one_group, batch_key="data", names=())
    with pytest.raises(ValueError, match="unique non-empty"):
        ResidualTerm(one_group, batch_key="data", names=("same", "same"))
    with pytest.raises(TypeError, match="callable"):
        ResidualTerm(object(), batch_key="data", names=("loss",))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="ntk_stream"):
        ResidualTerm(one_group, batch_key="data", names=("loss",), ntk_stream="invalid")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="default_ntk_stream='output'"):
        ResidualTerm(one_group, batch_key="data", names=("loss",), ntk_stream="output")
    with pytest.raises(ValueError, match="does not define residual names"):
        ResidualTerm(residual_fn=one_group, batch_key="data")

    term = ResidualTerm(one_group, batch_key="data", names=("first", "second"))
    with pytest.raises(ValueError, match="returned 1 groups"):
        term.losses(lambda state, point: point, None, {"data": {}})
    with pytest.raises(KeyError, match="Unknown objective stream"):
        term.residual_stream("missing", lambda state, point: point, None, {"data": {}})


def _returning(output: Any) -> Any:
    """Build an equation that returns a fixed residual output.

    Args:
        output: Value returned by the equation.

    Returns:
        Equation callable ignoring its model, state, batch, and stream.
    """

    def equation(model_apply: ModelApply, model_state: Any, batch: ArrayMapping, *, stream: ResidualStream) -> Any:
        """Return the fixed residual output.

        Args:
            model_apply: Unused model application callable.
            model_state: Unused model state.
            batch: Unused batch.
            stream: Unused requested stream.

        Returns:
            The fixed residual output.
        """
        del model_apply, model_state, batch, stream
        return output

    return equation


@pytest.mark.parametrize(
    ("output", "names", "expected_losses"),
    [
        (jnp.asarray([1.0, 3.0]), ("a",), {"a": 5.0}),
        ((jnp.asarray([1.0, 3.0]), jnp.asarray([[2.0], [2.0]])), ("a", "b"), {"a": 5.0, "b": 4.0}),
        ((jnp.asarray([2.0]), (jnp.asarray([1.0]), jnp.asarray([3.0]))), ("a", "b"), {"a": 4.0, "b": 10.0}),
    ],
    ids=["bare-array", "tuple-of-arrays", "array-and-group"],
)
def test_residual_term_accepts_bare_arrays_as_single_residual_groups(
    output: Any, names: tuple[str, ...], expected_losses: dict[str, float]
) -> None:
    """Verify a bare array is one loss, and a tuple entry is one loss whether it is an array or a residual group.

    Args:
        output: Equation output under test.
        names: Loss names aligned with the output entries.
        expected_losses: Mean-squared loss for each name.
    """
    term = ResidualTerm(_returning(output), batch_key="data", names=names)

    losses = term.losses(lambda state, point: point, None, {"data": {}})
    stream = term.residual_stream(names[0], lambda state, point: point, None, {"data": {}})

    assert {name: float(value) for name, value in losses.items()} == pytest.approx(expected_losses)
    assert stream.ndim == 2


def test_residual_term_rejects_outputs_that_are_not_arrays_or_tuples() -> None:
    """Verify a list or mapping output fails with the accepted return forms in the error."""
    for output in ([jnp.ones(2)], {"a": jnp.ones(2)}):
        term = ResidualTerm(_returning(output), batch_key="data", names=("a",))
        with pytest.raises(TypeError, match="an array, or a tuple of arrays and residual tuples"):
            term.losses(lambda state, point: point, None, {"data": {}})
