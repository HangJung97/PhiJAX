import jax
import jax.numpy as jnp


def apply_dropout(inputs: jax.Array, rate: float, key: jax.Array) -> jax.Array:
    """Apply inverted dropout using an explicit PRNG key.

    Args:
        inputs: Activation array.
        rate: Probability of dropping each activation.
        key: Explicit JAX PRNG key.

    Returns:
        Dropped and rescaled activations with the same shape as `inputs`.

    Raises:
        ValueError: If `rate` is outside `[0, 1)`.
    """
    if not 0.0 <= rate < 1.0:
        raise ValueError("`rate` must satisfy `0 <= rate < 1`.")
    keep_probability = 1.0 - rate
    mask = jax.random.bernoulli(key, keep_probability, inputs.shape)
    return jnp.where(mask, inputs / keep_probability, 0)
