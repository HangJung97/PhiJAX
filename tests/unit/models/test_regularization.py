import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.models.regularization import apply_dropout


def test_apply_dropout_is_reproducible_and_shape_preserving() -> None:
    """Verify explicit dropout keys produce deterministic repeated masks."""
    inputs = jnp.ones((16, 8), dtype=jnp.float32)

    first = apply_dropout(inputs, 0.5, jax.random.key(7))
    repeated = apply_dropout(inputs, 0.5, jax.random.key(7))

    np.testing.assert_array_equal(first, repeated)
    assert first.shape == inputs.shape


@pytest.mark.parametrize("rate", [-0.1, 1.0])
def test_apply_dropout_rejects_invalid_rates(rate: float) -> None:
    """Verify invalid dropout probabilities fail eagerly.

    Args:
        rate: Invalid dropout probability selected by pytest.
    """
    with pytest.raises(ValueError, match="0 <= rate < 1"):
        apply_dropout(jnp.ones(2), rate, jax.random.key(11))
