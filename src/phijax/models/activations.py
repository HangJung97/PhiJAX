from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Literal

import jax

Activation = Callable[[jax.Array], jax.Array]
ActivationName = Literal["relu", "leakyrelu", "gelu", "tanh", "sigmoid", "silu"]


def resolve_activation(
    activation: ActivationName | Activation | None,
    kwargs: Mapping[str, Any] | None = None,
) -> Activation | None:
    """Resolve an activation name or preserve a supplied callable.

    Args:
        activation: Supported activation name, callable, or `None`.
        kwargs: Keyword arguments bound to the resolved activation.

    Returns:
        An activation callable or `None`.

    Raises:
        ValueError: If `activation` is an unknown string.
    """
    if activation is None:
        return None
    activation_kwargs = dict(kwargs or {})
    if callable(activation):
        return partial(activation, **activation_kwargs)
    activations: dict[str, Activation] = {
        "relu": jax.nn.relu,
        "leakyrelu": jax.nn.leaky_relu,
        "gelu": jax.nn.gelu,
        "tanh": jax.nn.tanh,
        "sigmoid": jax.nn.sigmoid,
        "silu": jax.nn.silu,
    }
    if activation not in activations:
        choices = ", ".join(activations)
        raise ValueError(f"Unknown activation `{activation}`. Available activations: {choices}.")
    if activation == "gelu":
        activation_kwargs.setdefault("approximate", False)
    return partial(activations[activation], **activation_kwargs)
