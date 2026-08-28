import jax
import jax.numpy as jnp
import pytest
from flax import nnx
from rich.text import Text

from phijax.models import MLP, build_mlp, tabulate_nnx_model


def test_tabulate_nnx_model_reports_shapes_layers_and_parameter_count() -> None:
    """Verify the NNX summary reconstructs explicit state and traces representative inputs."""
    initialized = build_mlp(
        jax.random.key(12),
        input_dim=3,
        output_dim=2,
        input_mean=jnp.zeros(3),
        input_std=jnp.ones(3),
        hidden=(4,),
    )
    assert initialized.summary is not None
    summary = initialized.summary(initialized.state, console_width=120)
    plain_summary = Text.from_ansi(summary).plain

    assert "MLP Summary" in plain_summary
    assert "networks/0/0" in plain_summary
    assert "float32[1,3]" in plain_summary
    assert "float32[1,2]" in plain_summary
    assert "Total Parameters: 26" in plain_summary


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"max_depth": -2}, "max_depth"),
        ({"console_width": 0}, "console_width"),
    ],
)
def test_tabulate_nnx_model_rejects_invalid_display_policy(kwargs: dict[str, int], match: str) -> None:
    """Verify invalid summary display settings fail before graph reconstruction.

    Args:
        kwargs: Invalid summary keyword arguments.
        match: Expected error-message fragment.
    """
    model = MLP(1, 1, hidden=(), rngs=nnx.Rngs(params=jax.random.key(13)))
    graphdef, model_state = nnx.split(model)
    with pytest.raises(ValueError, match=match):
        tabulate_nnx_model(
            graphdef,
            model_state,
            example_inputs=jnp.zeros((1, 1), dtype=jnp.float32),
            **kwargs,
        )
