import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from rich.text import Text

from phijax.models import InitializedModel, initialize_nnx_model


class AffineModel(nnx.Module):
    """Provide a minimal custom NNX architecture for adapter tests.

    Attributes:
        linear: Trainable affine transformation.
    """

    def __init__(self, input_dim: int, output_dim: int, *, rngs: nnx.Rngs) -> None:
        """Initialize the custom architecture.

        Args:
            input_dim: Width of the input vector.
            output_dim: Width of the output vector.
            rngs: NNX random streams used for parameter initialization.
        """
        self.linear = nnx.Linear(input_dim, output_dim, rngs=rngs)

    def __call__(self, inputs: jax.Array, *, scale: float = 1.0) -> jax.Array:
        """Apply the affine transformation and an externally configured scale.

        Args:
            inputs: Input array with final width matching the affine layer.
            scale: Multiplicative output scale.

        Returns:
            Scaled affine predictions.
        """
        return scale * self.linear(inputs)


def test_initialize_nnx_model_adapts_custom_architecture() -> None:
    """Verify custom NNX modules receive pure application, state, and summary contracts."""
    model = AffineModel(2, 1, rngs=nnx.Rngs(params=jax.random.key(7)))
    initialized = initialize_nnx_model(
        model,
        example_inputs=jnp.zeros((1, 2), dtype=jnp.float32),
        call_kwargs={"scale": 2.0},
    )
    inputs = jnp.ones((3, 2), dtype=jnp.float32)

    outputs = jax.jit(initialized.apply)(initialized.state, inputs)
    gradients = jax.grad(lambda state: jnp.sum(initialized.apply(state, inputs)))(initialized.state)

    assert isinstance(initialized, InitializedModel)
    assert outputs.shape == (3, 1)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(gradients))
    assert initialized.summary is not None
    assert "AffineModel Summary" in Text.from_ansi(initialized.summary(initialized.state)).plain


def test_initialize_nnx_model_allows_per_call_keyword_overrides() -> None:
    """Verify dynamic call keywords take precedence over factory-bound defaults."""
    model = AffineModel(1, 1, rngs=nnx.Rngs(params=jax.random.key(11)))
    initialized = initialize_nnx_model(model, call_kwargs={"scale": 2.0})
    inputs = jnp.ones((2, 1), dtype=jnp.float32)

    bound_outputs = initialized.apply(initialized.state, inputs)
    overridden_outputs = initialized.apply(initialized.state, inputs, scale=3.0)

    np.testing.assert_allclose(overridden_outputs, 1.5 * bound_outputs, rtol=1.0e-6)
    assert initialized.summary is None
