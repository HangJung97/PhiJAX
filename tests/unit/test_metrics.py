import jax
import jax.numpy as jnp

from phijax import TrainingOutput


def test_training_output_is_a_jittable_pytree() -> None:
    """Verify stable loss and diagnostic mappings survive a compiled transformation."""

    @jax.jit
    def evaluate(value: jax.Array) -> TrainingOutput:
        """Create one synthetic compiled training output.

        Args:
            value: Scalar input value.

        Returns:
            Stable loss and causal-style diagnostic mappings.
        """
        return TrainingOutput(
            losses={"pde/heat": value**2},
            diagnostics={
                "causal/mean_weight": value + 0.5,
                "causal/window_weights": jnp.stack((value, value + 1.0)),
            },
        )

    output = evaluate(jnp.asarray(2.0))
    assert float(output.losses["pde/heat"]) == 4.0
    assert float(output.diagnostics["causal/mean_weight"]) == 2.5
    assert output.diagnostics["causal/window_weights"].tolist() == [2.0, 3.0]
