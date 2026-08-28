import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from rich.text import Text

from phijax.models import FactorizedDense, ModifiedMLP, build_modified_mlp, initialize_nnx_model


def test_modified_mlp_matches_one_layer_gating_equation() -> None:
    """Verify one hidden layer implements the reference latent-gating equation exactly."""
    model = ModifiedMLP(2, 1, hidden_dim=6, num_layers=1, rngs=nnx.Rngs(params=jax.random.key(3)))
    inputs = jax.random.normal(jax.random.key(5), (4, 2))

    features = model.coordinate_features(inputs)
    gate_u = model.activation(model.gate_u(features))
    gate_v = model.activation(model.gate_v(features))
    hidden = model.activation(model.layers[0](features))
    expected = model.output_layer(hidden * gate_u + (1.0 - hidden) * gate_v)

    np.testing.assert_allclose(model(inputs), expected, rtol=1.0e-6, atol=1.0e-6)


def test_build_modified_mlp_supports_jit_gradients_and_summary() -> None:
    """Verify the Trainer-facing factory returns a compiled explicit-state model contract."""
    initialized = build_modified_mlp(
        jax.random.key(7),
        input_dim=2,
        output_dim=2,
        input_mean=jnp.asarray([0.5, -0.5]),
        input_std=jnp.asarray([0.25, 2.0]),
        hidden_dim=8,
        num_layers=3,
        input_norm=True,
        output_names=("u", "v"),
    )
    inputs = jnp.asarray([[0.5, 0.0], [1.0, -0.5]], dtype=jnp.float32)

    outputs = jax.jit(initialized.apply)(initialized.state, inputs)
    gradients = jax.grad(lambda state: jnp.sum(initialized.apply(state, inputs)))(initialized.state)

    assert outputs.shape == (2, 2)
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(gradients))
    assert initialized.summary is not None
    assert "ModifiedMLP Summary" in Text.from_ansi(initialized.summary(initialized.state)).plain


def test_modified_mlp_composes_periodic_fourier_and_factorized_layers() -> None:
    """Verify shared embedding and random weight-factorization policies compose with latent gating."""
    model = ModifiedMLP(
        2,
        1,
        hidden_dim=8,
        num_layers=2,
        periodic_features=True,
        periodic_features_kwargs={"axes": (1,), "frequencies": (jnp.pi,)},
        fourier_features=True,
        fourier_features_kwargs={"embed_dim": 4, "scale": 1.0},
        weight_factorization=True,
        rngs=nnx.Rngs(params=jax.random.key(11)),
    )
    inputs = jnp.asarray([[0.25, -1.0], [0.25, 1.0]], dtype=jnp.float32)

    outputs = model(inputs)

    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1.0e-5, atol=1.0e-6)
    assert isinstance(model.gate_u, FactorizedDense)
    assert all(isinstance(layer, FactorizedDense) for layer in model.layers)


def test_custom_modified_mlp_uses_generic_nnx_adapter_and_named_outputs() -> None:
    """Verify direct architecture use needs no specialized split helper and preserves output names."""
    model = ModifiedMLP(
        2,
        2,
        hidden_dim=8,
        num_layers=2,
        output_names=("velocity", "pressure"),
        rngs=nnx.Rngs(params=jax.random.key(13)),
    )
    initialized = initialize_nnx_model(model)

    outputs = initialized.apply(initialized.state, jnp.ones((3, 2), dtype=jnp.float32))

    assert outputs[model.slice_output("pressure")].shape == (3, 1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_layers": 0}, "must be positive"),
        ({"input_norm_eps": -1.0}, "non-negative"),
        ({"weight_norm": True, "weight_factorization": True}, "cannot both"),
        ({"output_names": ("duplicate", "duplicate")}, "unique"),
    ],
)
def test_modified_mlp_rejects_invalid_architecture_options(kwargs: dict[str, object], match: str) -> None:
    """Verify malformed widths, normalization, and dense policies fail eagerly.

    Args:
        kwargs: Invalid keyword arguments passed to :class:`ModifiedMLP`.
        match: Expected validation-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        ModifiedMLP(2, 2, rngs=nnx.Rngs(params=jax.random.key(17)), **kwargs)


def test_modified_mlp_validates_runtime_inputs_and_output_names() -> None:
    """Verify coordinate widths and unconfigured output-name access fail clearly."""
    model = ModifiedMLP(2, 1, hidden_dim=8, num_layers=1, rngs=nnx.Rngs(params=jax.random.key(19)))

    with pytest.raises(ValueError, match="final input width"):
        model(jnp.ones((2, 3), dtype=jnp.float32))
    with pytest.raises(ValueError, match="output_names"):
        model.slice_output("u")
