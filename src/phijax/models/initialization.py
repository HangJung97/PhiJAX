from collections.abc import Callable, Mapping
from functools import partial
from typing import Any, Literal

import jax

Initializer = Callable[[jax.Array, tuple[int, ...], Any], jax.Array]
InitializationName = Literal["kaiming_normal", "xavier_uniform", "trunc_normal"]


def resolve_initializer(
    initialization: InitializationName | Initializer | None,
    kwargs: Mapping[str, Any] | None = None,
) -> Initializer | None:
    """Resolve a dense-kernel initializer.

    Args:
        initialization: Supported initializer name, callable, or `None` for the NNX default.
        kwargs: Keyword arguments forwarded to the initializer factory.

    Returns:
        A JAX initializer callable or `None`.

    Raises:
        ValueError: If `initialization` is an unknown string.
    """
    if initialization is None or callable(initialization):
        return initialization
    initializer_kwargs = dict(kwargs or {})
    factories: dict[str, Callable[..., Initializer]] = {
        "kaiming_normal": partial(jax.nn.initializers.variance_scaling, 2.0, "fan_in", "truncated_normal"),
        "xavier_uniform": jax.nn.initializers.glorot_uniform,
        "trunc_normal": jax.nn.initializers.truncated_normal,
    }
    if initialization not in factories:
        choices = ", ".join(factories)
        raise ValueError(f"Unknown initialization `{initialization}`. Available initializations: {choices}.")
    return factories[initialization](**initializer_kwargs)
