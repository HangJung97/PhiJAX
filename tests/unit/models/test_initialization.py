import jax
import jax.numpy as jnp
import pytest

from phijax.models.initialization import resolve_initializer


@pytest.mark.parametrize("name", ["kaiming_normal", "xavier_uniform", "trunc_normal"])
def test_resolve_initializer_builds_finite_float32_weights(name: str) -> None:
    """Verify every configured initializer builds finite weights.

    Args:
        name: Registered initializer name selected by pytest.
    """
    initializer = resolve_initializer(name)  # type: ignore[arg-type]

    assert initializer is not None
    weights = initializer(jax.random.key(3), (4, 5), jnp.float32)
    assert weights.shape == (4, 5)
    assert weights.dtype == jnp.float32
    assert bool(jnp.all(jnp.isfinite(weights)))


def test_resolve_initializer_preserves_callable_and_none_values() -> None:
    """Verify custom initializers and default-NNX selection pass through unchanged."""
    assert resolve_initializer(jax.nn.initializers.ones) is jax.nn.initializers.ones
    assert resolve_initializer(None) is None


def test_resolve_initializer_rejects_an_unknown_name() -> None:
    """Verify unknown initializer names fail before layer construction."""
    with pytest.raises(ValueError, match="Unknown initialization"):
        resolve_initializer("unknown")  # type: ignore[arg-type]
