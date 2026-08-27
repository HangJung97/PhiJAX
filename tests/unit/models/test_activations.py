import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.models.activations import resolve_activation


@pytest.mark.parametrize("name", ["relu", "leakyrelu", "gelu", "tanh", "sigmoid", "silu"])
def test_resolve_activation_supports_each_configured_name(name: str) -> None:
    """Verify every configured activation produces finite shape-preserving values.

    Args:
        name: Registered activation name selected by pytest.
    """
    activation = resolve_activation(name)  # type: ignore[arg-type]

    assert activation is not None
    output = activation(jnp.asarray([-1.0, 0.0, 1.0], dtype=jnp.float32))
    assert output.shape == (3,)
    assert bool(jnp.all(jnp.isfinite(output)))


def test_resolve_activation_uses_exact_gelu_by_default() -> None:
    """Verify GELU defaults use the non-approximate implementation."""
    inputs = jnp.asarray([-0.5, 0.5], dtype=jnp.float32)
    activation = resolve_activation("gelu")

    assert activation is not None
    np.testing.assert_allclose(activation(inputs), jax.nn.gelu(inputs, approximate=False), rtol=1.0e-6)


def test_resolve_activation_binds_kwargs_to_a_callable() -> None:
    """Verify custom activation callables receive configured keyword arguments."""

    def scale(inputs: jax.Array, *, factor: float) -> jax.Array:
        """Scale activation inputs by a configured factor.

        Args:
            inputs: Activation input array.
            factor: Multiplicative scale.

        Returns:
            Scaled activation array.
        """
        return factor * inputs

    activation = resolve_activation(scale, {"factor": 2.0})

    assert activation is not None
    np.testing.assert_array_equal(activation(jnp.ones(2)), np.full(2, 2.0))


def test_resolve_activation_rejects_an_unknown_name() -> None:
    """Verify unknown activation names fail before model construction."""
    with pytest.raises(ValueError, match="Unknown activation"):
        resolve_activation("unknown")  # type: ignore[arg-type]
