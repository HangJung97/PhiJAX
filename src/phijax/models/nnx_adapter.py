from collections.abc import Mapping
from functools import partial
from typing import Any

import jax
from flax import nnx

from phijax.models.contracts import InitializedModel
from phijax.models.summary import tabulate_nnx_model


def initialize_nnx_model(
    model: nnx.Module,
    *,
    example_inputs: jax.Array | None = None,
    call_kwargs: Mapping[str, Any] | None = None,
) -> InitializedModel:
    """Adapt an initialized Flax NNX module to PhiJAX's explicit-state model contract.

    The adapter separates the static NNX graph from its array state and captures only static call configuration in the
    returned application callable. Architecture factories remain responsible for constructing `model` with an explicit
    PRNG key and choosing representative `example_inputs`.

    Args:
        model: Initialized NNX module to split into a graph definition and array state.
        example_inputs: Optional representative inputs used to provide a model-summary callable.
        call_kwargs: Optional keyword arguments bound to every model call, such as normalization statistics.

    Returns:
        Pure model application, explicit NNX state, and an optional model-summary callable.
    """
    graphdef, model_state = nnx.split(model)
    bound_call_kwargs = dict(call_kwargs or {})

    def apply_model(state: nnx.State, inputs: jax.Array, *args: Any, **kwargs: Any) -> jax.Array:
        """Apply the split NNX graph with explicit state.

        Args:
            state: NNX array state compatible with the captured graph definition.
            inputs: Model input array.
            *args: Additional positional arguments forwarded to the module.
            **kwargs: Per-call keyword arguments overriding any bound `call_kwargs`.

        Returns:
            Output produced by the reconstructed NNX module.
        """
        resolved_kwargs = bound_call_kwargs | kwargs
        return nnx.merge(graphdef, state)(inputs, *args, **resolved_kwargs)

    summary = None
    if example_inputs is not None:
        summary = partial(tabulate_nnx_model, graphdef, example_inputs=example_inputs)
    return InitializedModel(apply_model, model_state, summary)


__all__ = ["initialize_nnx_model"]
