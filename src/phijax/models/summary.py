from typing import Any

import jax
from flax import nnx


def tabulate_nnx_model(
    graphdef: Any,
    model_state: Any,
    *,
    example_inputs: jax.Array,
    max_depth: int = -1,
    console_width: int = 120,
    compute_flops: bool = False,
    compute_vjp_flops: bool = False,
) -> str:
    """Render a Flax NNX graph and explicit state with :func:`flax.nnx.tabulate`.

    Args:
        graphdef: Static NNX graph definition paired with `model_state`.
        model_state: Explicit NNX variable state compatible with `graphdef`.
        example_inputs: Representative batched model inputs used to trace shapes.
        max_depth: Maximum displayed module depth, or `-1` for every level.
        console_width: Positive Rich console width in characters.
        compute_flops: Whether to estimate forward-pass floating-point operations.
        compute_vjp_flops: Whether to estimate reverse-pass floating-point operations.

    Returns:
        Rich-formatted model-summary table.

    Raises:
        ValueError: If `max_depth` is below `-1` or `console_width` is not positive.
    """
    if max_depth < -1:
        raise ValueError("`max_depth` must be `-1` or nonnegative.")
    if console_width < 1:
        raise ValueError("`console_width` must be positive.")
    model = nnx.merge(graphdef, model_state)
    return nnx.tabulate(
        model,
        example_inputs,
        depth=None if max_depth == -1 else max_depth,
        console_kwargs={"width": console_width},
        compute_flops=compute_flops,
        compute_vjp_flops=compute_vjp_flops,
    )


__all__ = ["tabulate_nnx_model"]
