from typing import Any

import jax
import jax.numpy as jnp
import pytest

from phijax.objectives import CompositeObjective, Objective, ObjectiveTerm, ResidualObjective
from phijax.types import ModelApply, NamedBatches


class _ScalarObjective:
    """Implement the generic scalar-loss objective contract structurally."""

    loss_names = ("data",)
    batch_keys = ("data",)

    def losses(
        self,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> dict[str, jax.Array]:
        """Return one synthetic scalar loss.

        Args:
            model_apply: Unused model application callable.
            model_state: Unused model state.
            batches: Unused named batches.

        Returns:
            Mapping containing one scalar loss.
        """
        del model_apply, model_state, batches
        return {"data": jnp.asarray(1.0)}


class _StreamObjective(_ScalarObjective):
    """Extend the synthetic objective with a raw residual stream."""

    def residual_stream(
        self,
        name: str,
        model_apply: ModelApply,
        model_state: Any,
        batches: NamedBatches,
    ) -> jax.Array:
        """Return one synthetic residual stream.

        Args:
            name: Requested loss name.
            model_apply: Unused model application callable.
            model_state: Unused model state.
            batches: Unused named batches.

        Returns:
            One-element residual array.
        """
        del name, model_apply, model_state, batches
        return jnp.ones(1)


def test_objective_protocols_accept_structural_implementations() -> None:
    """Verify applications can implement objective contracts without inheriting framework base classes."""
    assert isinstance(_ScalarObjective(), Objective)
    assert not isinstance(_ScalarObjective(), ResidualObjective)
    assert isinstance(_StreamObjective(), Objective)
    assert isinstance(_StreamObjective(), ResidualObjective)
    assert isinstance(_StreamObjective(), ObjectiveTerm)


def test_composite_objective_validates_terms_and_loss_names() -> None:
    """Verify declarative compositions require terms with globally unique names."""
    with pytest.raises(ValueError, match="at least one"):
        CompositeObjective({})
    with pytest.raises(TypeError, match="ObjectiveTerm"):
        CompositeObjective({"invalid": object()})  # type: ignore[dict-item]
    with pytest.raises(ValueError, match="globally unique"):
        CompositeObjective({"first": _StreamObjective(), "second": _StreamObjective()})


def test_composite_objective_reports_the_batch_keys_of_each_stream() -> None:
    """Verify each stream maps to its owning term's batch keys, and the module forwards them."""
    from functools import partial

    from phijax import PhiModule
    from phijax.equations import base_data_fidelity
    from phijax.models import build_mlp

    objective = CompositeObjective.from_equations({"data": base_data_fidelity, "anchor": base_data_fidelity})
    data_name, anchor_name = objective.loss_names

    assert objective.stream_batch_keys(data_name) == ("data",)
    assert objective.stream_batch_keys(anchor_name) == ("anchor",)
    with pytest.raises(KeyError, match="Unknown objective stream"):
        objective.stream_batch_keys("missing")

    module = PhiModule(partial(build_mlp, input_dim=1, output_dim=1, hidden=(4,)), objective)
    assert module.stream_batch_keys(anchor_name) == ("anchor",)
