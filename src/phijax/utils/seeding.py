import random
import secrets

import jax
import numpy as np


def resolve_seed(seed: int | None) -> int:
    """Resolve an optional root seed to a concrete unsigned 32-bit integer.

    A configured integer is preserved for reproducibility. Passing `None` draws one seed from the operating system's
    cryptographic random source so it does not depend on Python or NumPy RNG state that this seed will subsequently
    initialize.

    Args:
        seed: Configured root seed, or `None` to generate one randomly.

    Returns:
        Concrete unsigned 32-bit root seed.

    Raises:
        TypeError: If `seed` is neither an integer nor `None`, or is Boolean.
        ValueError: If an integer seed lies outside the unsigned 32-bit range.
    """
    if seed is None:
        return secrets.randbits(32)
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("`seed` must be an integer or `None`.")
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("`seed` must be an unsigned 32-bit integer.")
    return seed


def seed_everything(seed: int, *, process_index: int | None = None) -> jax.Array:
    """Globally seed Python and NumPy and return an explicit process-local JAX key.

    JAX has no implicit global random generator, so callers must still thread the returned key through model
    initialization, sampling, and functional training state. In distributed execution, host RNGs and the returned JAX
    key are deterministically separated by process index.

    Args:
        seed: Reproducible unsigned 32-bit root seed.
        process_index: Optional global JAX process index. Defaults to :func:`jax.process_index`.

    Returns:
        Process-local typed JAX PRNG key derived from `seed`.

    Raises:
        ValueError: If `seed` or `process_index` is outside its valid range.
    """
    if not 0 <= seed <= np.iinfo(np.uint32).max:
        raise ValueError("`seed` must be an unsigned 32-bit integer.")
    resolved_process_index = jax.process_index() if process_index is None else process_index
    if resolved_process_index < 0:
        raise ValueError("`process_index` must be non-negative.")

    # SeedSequence avoids simple adjacent-seed correlations between distributed host-side samplers.
    host_seed = int(np.random.SeedSequence((seed, resolved_process_index)).generate_state(1, dtype=np.uint32)[0])
    random.seed(host_seed)
    np.random.seed(host_seed)
    return jax.random.fold_in(jax.random.key(seed), resolved_process_index)


__all__ = ["resolve_seed", "seed_everything"]
