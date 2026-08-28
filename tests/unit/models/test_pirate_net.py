import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from rich.text import Text

from phijax.models import FactorizedDense, PirateNet, build_pirate_net, initialize_nnx_model


def test_pirate_net_blocks_are_exact_identities_at_zero_initialization() -> None:
    """Verify zero adaptive coefficients reduce every residual block to the identity map."""
    model = PirateNet(2, 1, hidden_dim=8, num_blocks=3, rngs=nnx.Rngs(params=jax.random.key(3)))
    inputs = jax.random.normal(jax.random.key(5), (4, 2))

    features = model.coordinate_features(inputs)
    expected = model.output_layer(features)
    outputs = model(inputs)

    np.testing.assert_allclose(outputs, expected, rtol=1.0e-6, atol=1.0e-6)
    assert all(float(block.alpha[...]) == 0.0 for block in model.blocks)


def test_pirate_net_adaptive_coefficient_activates_nonlinear_block() -> None:
    """Verify a nonzero adaptive coefficient changes the initially shallow forward pass."""
    model = PirateNet(2, 1, hidden_dim=8, num_blocks=1, rngs=nnx.Rngs(params=jax.random.key(7)))
    inputs = jax.random.normal(jax.random.key(11), (4, 2))
    shallow_outputs = model(inputs)

    model.blocks[0].alpha[...] = jnp.asarray(1.0)
    nonlinear_outputs = model(inputs)

    assert not bool(jnp.allclose(shallow_outputs, nonlinear_outputs))


def test_build_pirate_net_supports_jit_gradients_names_and_summary() -> None:
    """Verify the Trainer-facing factory provides the complete explicit-state model contract."""
    initialized = build_pirate_net(
        jax.random.key(13),
        input_dim=3,
        output_dim=2,
        input_mean=jnp.asarray([0.5, -0.5, 1.0]),
        input_std=jnp.asarray([0.5, 2.0, 1.5]),
        hidden_dim=12,
        num_blocks=2,
        input_norm=True,
        output_names=("velocity", "pressure"),
    )
    inputs = jnp.asarray([[0.5, 0.0, 1.5], [1.0, -0.5, 0.0]], dtype=jnp.float32)

    outputs = jax.jit(initialized.apply)(initialized.state, inputs)
    gradients = jax.grad(lambda state: jnp.sum(initialized.apply(state, inputs)))(initialized.state)

    assert outputs.shape == (2, 2)
    assert outputs.dtype == jnp.float32
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(gradients))
    assert initialized.summary is not None
    assert "PirateNet Summary" in Text.from_ansi(initialized.summary(initialized.state)).plain


def test_pirate_net_supports_periodic_features_and_named_output_slices() -> None:
    """Verify periodic preprocessing composes with Fourier features and named scalar outputs."""
    model = PirateNet(
        2,
        2,
        hidden_dim=8,
        num_blocks=1,
        output_names=("u", "v"),
        periodic_features=True,
        periodic_features_kwargs={"axes": (1,), "frequencies": (jnp.pi,)},
        rngs=nnx.Rngs(params=jax.random.key(17)),
    )
    inputs = jnp.asarray([[0.25, -1.0], [0.25, 1.0]], dtype=jnp.float32)

    outputs = model(inputs)

    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1.0e-5, atol=1.0e-6)
    assert outputs[model.slice_output("v")].shape == (2, 1)


def test_pirate_net_supports_random_weight_factorization() -> None:
    """Verify all affine transformations can use the paper's recommended random weight factorization."""
    model = PirateNet(
        2,
        1,
        hidden_dim=8,
        num_blocks=2,
        weight_factorization=True,
        weight_factorization_kwargs={"mean": 0.5, "std": 0.1},
        rngs=nnx.Rngs(params=jax.random.key(19)),
    )

    outputs = model(jnp.ones((3, 2), dtype=jnp.float32))

    assert isinstance(model.gate_u, FactorizedDense)
    assert isinstance(model.gate_v, FactorizedDense)
    assert isinstance(model.output_layer, FactorizedDense)
    assert all(isinstance(layer, FactorizedDense) for block in model.blocks for layer in block.layers)
    assert bool(jnp.all(jnp.isfinite(outputs)))


def test_custom_pirate_net_uses_generic_nnx_adapter() -> None:
    """Verify direct architecture construction needs no architecture-specific split or apply helper."""
    model = PirateNet(2, 1, hidden_dim=8, num_blocks=1, rngs=nnx.Rngs(params=jax.random.key(23)))
    initialized = initialize_nnx_model(model)

    outputs = initialized.apply(initialized.state, jnp.ones((2, 2), dtype=jnp.float32))

    assert outputs.shape == (2, 1)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"num_blocks": 0}, "must be positive"),
        ({"hidden_dim": 7}, "even"),
        ({"hidden_dim": 8, "fourier_features_kwargs": {"embed_dim": 3}}, "must have width"),
        ({"hidden_dim": 8, "fourier_features": False}, "must have width"),
        ({"hidden_dim": 8, "num_blocks": 2, "nonlinearity": (0.0,)}, "must contain 2"),
        ({"hidden_dim": 8, "nonlinearity": float("nan")}, "finite"),
        ({"hidden_dim": 8, "output_names": ("duplicate", "duplicate")}, "unique"),
    ],
)
def test_pirate_net_rejects_invalid_architecture_options(kwargs: dict[str, object], match: str) -> None:
    """Verify malformed embedding and residual policies fail during construction.

    Args:
        kwargs: Invalid keyword arguments passed to :class:`PirateNet`.
        match: Expected validation-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        PirateNet(2, 2, rngs=nnx.Rngs(params=jax.random.key(29)), **kwargs)


def test_pirate_net_validates_inputs_and_named_output_access() -> None:
    """Verify runtime coordinate shapes and named-output queries fail clearly."""
    model = PirateNet(2, 1, hidden_dim=8, num_blocks=1, rngs=nnx.Rngs(params=jax.random.key(31)))

    with pytest.raises(ValueError, match="final input width"):
        model(jnp.ones((2, 3), dtype=jnp.float32))
    with pytest.raises(ValueError, match="output_names"):
        model.slice_output("u")
