from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx
from omegaconf import OmegaConf

from phijax.models import (
    MLP,
    FactorizedDense,
    InitializedModel,
    PeriodicFeatures,
    build_mlp,
    build_modified_mlp,
    build_pirate_net,
    initialize_nnx_model,
)


@pytest.mark.parametrize(
    ("builder", "architecture"),
    [
        (build_mlp, {"hidden": (4,)}),
        (build_modified_mlp, {"hidden_dim": 4, "num_layers": 1}),
        (build_pirate_net, {"hidden_dim": 4, "num_blocks": 1}),
    ],
)
def test_model_builders_skip_optional_input_normalization(
    builder: Callable[..., InitializedModel],
    architecture: dict[str, object],
) -> None:
    """Verify built-in factories work when a DataModule supplies no input statistics.

    Args:
        builder: Built-in initialized-model factory.
        architecture: Small architecture options for the selected factory.
    """
    initialized = builder(jax.random.key(0), input_dim=2, output_dim=1, **architecture)

    outputs = initialized.apply(initialized.state, jnp.ones((3, 2), dtype=jnp.float32))

    assert outputs.shape == (3, 1)
    assert bool(jnp.all(jnp.isfinite(outputs)))


def test_factorized_dense_reconstructs_finite_weight() -> None:
    """Verify factorized parameter layouts and finite affine outputs."""
    layer = FactorizedDense(3, 4, rngs=nnx.Rngs(7))
    output = layer(jnp.ones((2, 3), dtype=jnp.float32))

    assert output.shape == (2, 4)
    assert layer.g[...].shape == (4,)
    assert layer.v[...].shape == (3, 4)
    assert layer.bias is not None
    assert bool(jnp.all(jnp.isfinite(output)))


def test_factorized_dense_supports_biasless_custom_initialization() -> None:
    """Verify configurable base-weight initialization and optional bias."""
    layer = FactorizedDense(
        2,
        3,
        use_bias=False,
        kernel_init=jax.nn.initializers.ones,
        rngs=nnx.Rngs(2),
    )

    output = layer(jnp.ones((1, 2), dtype=jnp.float32))

    assert layer.bias is None
    np.testing.assert_allclose(output, np.full((1, 3), 2.0, dtype=np.float32), rtol=1.0e-6)


def test_periodic_features_enforce_value_and_derivative_periodicity() -> None:
    """Verify a physical period identifies both ends of a periodic spatial interval."""
    embedding = PeriodicFeatures(2, periods={1: 2.0})
    inputs = jnp.asarray([[0.25, -1.0], [0.25, 1.0]], dtype=jnp.float32)

    features = embedding(inputs)

    assert features.shape == (2, 3)
    np.testing.assert_allclose(features[0], features[1], rtol=1e-6, atol=2e-7)

    def embedded_space(position: jax.Array) -> jax.Array:
        """Embed one fixed-time point as a function of space.

        Args:
            position: Scalar spatial coordinate.

        Returns:
            Periodically embedded coordinate vector.
        """
        return embedding(jnp.stack((jnp.asarray(0.25), position)))

    lower_derivative = jax.jacfwd(embedded_space)(jnp.asarray(-1.0))
    upper_derivative = jax.jacfwd(embedded_space)(jnp.asarray(1.0))
    np.testing.assert_allclose(lower_derivative, upper_derivative, rtol=1e-6, atol=1e-6)
    with pytest.raises(ValueError, match="final input width"):
        embedding(jnp.ones((2, 3), dtype=jnp.float32))


def test_mlp_composes_periodic_and_random_fourier_features() -> None:
    """Verify the Burgers embedding chain preserves periodic equality and reconciles intermediate widths."""
    model = MLP(
        2,
        1,
        hidden=(4,),
        periodic_features=True,
        periodic_features_kwargs={"periods": {1: 2.0}},
        fourier_features=True,
        fourier_features_kwargs={"embed_dim": 3, "scale": 2.0},
        rngs=nnx.Rngs(params=jax.random.key(17)),
    )
    initialized = initialize_nnx_model(model)
    inputs = jnp.asarray([[0.25, -1.0], [0.25, 1.0]], dtype=jnp.float32)

    outputs = initialized.apply(initialized.state, inputs)

    assert outputs.shape == (2, 1)
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"periods": {}}, "at least one"),
        ({"periods": {2: 1.0}}, "valid input coordinate"),
        ({"periods": {1: 1.0, "1": 2.0}}, "unique"),
        ({"periods": {-1: 1.0}}, "nonnegative"),
        ({"periods": {1: 0.0}}, "finite and positive"),
        ({"periods": {1: float("inf")}}, "finite and positive"),
    ],
)
def test_periodic_features_reject_invalid_configuration(kwargs: dict[str, object], match: str) -> None:
    """Verify invalid periodic coordinate policies fail during model construction.

    Args:
        kwargs: Invalid periodic-feature options.
        match: Expected validation-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        PeriodicFeatures(2, **kwargs)


def test_mlp_split_apply_and_gradients_with_polar_options() -> None:
    """Verify Fourier-factorized application and gradients through split NNX state."""
    model = MLP(
        2,
        4,
        hidden=(8, 8),
        activation="gelu",
        input_norm=True,
        fourier_features=True,
        fourier_features_kwargs={"embed_dim": 4, "scale": 1.0},
        weight_factorization=True,
        weight_factorization_kwargs={"mean": 0.5, "std": 0.1},
        rngs=nnx.Rngs(params=jax.random.key(11)),
    )
    initialized = initialize_nnx_model(
        model,
        call_kwargs={"input_mean": jnp.zeros(2), "input_std": jnp.ones(2)},
    )
    inputs = jnp.asarray([[0.5, 0.2], [0.8, -0.1]], dtype=jnp.float32)

    def summed_output(model_state: nnx.State) -> jax.Array:
        """Reduce model outputs to a differentiable scalar.

        Args:
            model_state: Explicit NNX array state to differentiate.

        Returns:
            Sum of all outputs for the fixed test coordinates.
        """
        return jnp.sum(initialized.apply(model_state, inputs))

    output = initialized.apply(initialized.state, inputs)
    gradients = jax.grad(summed_output)(initialized.state)

    assert output.shape == (2, 4)
    assert output.dtype == jnp.float32
    assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(gradients))


def test_mlp_supports_empty_hidden_layout_and_named_outputs() -> None:
    """Verify a direct dense model and stable named-output slicing."""
    model = MLP(
        1,
        2,
        hidden=None,
        activation=None,
        output_activation="sigmoid",
        output_names=("velocity", "pressure"),
        rngs=nnx.Rngs(3),
    )

    output = model(jnp.ones((5, 1)))

    assert output.shape == (5, 2)
    assert bool(jnp.all((output >= 0.0) & (output <= 1.0)))
    assert output[model.slice_output("pressure")].shape == (5, 1)


def test_mlp_resolves_hydra_style_compute_dtype_strings() -> None:
    """Verify string dtypes from OmegaConf are resolved for JAX arithmetic."""
    model = MLP(2, 1, hidden=(3,), compute_dtype="bfloat16", rngs=nnx.Rngs(4))

    output = model(jnp.ones((2, 2), dtype=jnp.float32))

    assert model.compute_dtype == jnp.dtype(jnp.bfloat16)
    assert output.dtype == jnp.float32


def test_mlp_supports_one_network_per_output_and_weight_normalization() -> None:
    """Verify independent scalar branches and NNX weight-normalized dense layers."""
    model = MLP(
        3,
        2,
        hidden=(4,),
        one_mlp_per_output=True,
        weight_norm=True,
        rngs=nnx.Rngs(5),
    )

    output = model(jnp.ones((2, 3), dtype=jnp.float32))

    assert len(model.networks) == 2
    assert output.shape == (2, 2)
    assert bool(jnp.all(jnp.isfinite(output)))


def test_mlp_dropout_is_explicit_and_reproducible() -> None:
    """Verify stochastic dropout requires a key and repeats for an identical key."""
    model = MLP(3, 2, hidden=(16,), dropout=0.5, rngs=nnx.Rngs(7))
    inputs = jnp.ones((8, 3), dtype=jnp.float32)

    with pytest.raises(ValueError, match="dropout_key"):
        model(inputs, deterministic=False)

    first = model(inputs, deterministic=False, dropout_key=jax.random.key(19))
    repeated = model(inputs, deterministic=False, dropout_key=jax.random.key(19))
    different = model(inputs, deterministic=False, dropout_key=jax.random.key(23))

    np.testing.assert_array_equal(first, repeated)
    assert not bool(jnp.array_equal(first, different))


def test_mlp_clamps_singleton_input_standard_deviation() -> None:
    """Verify finite optional normalization with a zero standard deviation."""
    model = MLP(3, 3, hidden=(4,), input_norm=True, rngs=nnx.Rngs(params=jax.random.key(5)))
    output = model(jnp.ones((1, 3)), jnp.ones(3), jnp.zeros(3))
    assert bool(jnp.all(jnp.isfinite(output)))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"dropout": 1.0}, "dropout"),
        ({"weight_norm": True, "weight_factorization": True}, "cannot both"),
        ({"output_names": ("duplicate", "duplicate")}, "unique"),
    ],
)
def test_mlp_rejects_invalid_dynamic_options(kwargs: dict[str, object], message: str) -> None:
    """Verify incompatible or malformed architecture options fail eagerly.

    Args:
        kwargs: Invalid keyword arguments passed to :class:`MLP`.
        message: Expected validation error fragment.
    """
    with pytest.raises(ValueError, match=message):
        MLP(2, 2, rngs=nnx.Rngs(29), **kwargs)


def test_periodic_features_use_physical_coordinates() -> None:
    """Verify periodic axes embed `2*pi*x/T` of the physical coordinate and other axes use the normalized values."""
    period = 0.84
    embedding = PeriodicFeatures(2, periods={1: period})
    inputs = jnp.asarray([[0.3, 10.3], [-1.2, 0.05]], dtype=jnp.float32)
    normalized = ((inputs - 1.0) / 3.0).astype(jnp.bfloat16)

    features = embedding(inputs, normalized)
    phase = 2.0 * np.pi * np.asarray(inputs[:, 1], np.float64) / period

    assert embedding.periods == ((1, period),)
    assert features.shape == (2, 3)
    np.testing.assert_array_equal(features[:, 0], normalized[:, 0].astype(jnp.float32))
    # The phase comes from the float32 physical inputs, not from the low-precision normalized values.
    np.testing.assert_allclose(features[:, 1], np.cos(phase), atol=2e-5)
    np.testing.assert_allclose(features[:, 2], np.sin(phase), atol=2e-5)
    np.testing.assert_array_equal(embedding(inputs)[:, 0], inputs[:, 0])

    def embedded_time(time: jax.Array) -> jax.Array:
        """Embed one point as a function of its periodic coordinate.

        Args:
            time: Scalar periodic coordinate.

        Returns:
            Embedded feature vector.
        """
        return embedding(jnp.stack((jnp.asarray(0.3), time)))

    derivative = jax.jacfwd(embedded_time)(jnp.asarray(0.25))
    angle = 2.0 * np.pi * 0.25 / period
    np.testing.assert_allclose(
        derivative[1:], 2.0 * np.pi / period * np.asarray([-np.sin(angle), np.cos(angle)]), rtol=1e-5
    )


@pytest.mark.parametrize("input_norm", [True, False], ids=["normalized", "raw"])
@pytest.mark.parametrize("builder", [build_mlp, build_modified_mlp, build_pirate_net])
def test_models_repeat_every_declared_period_for_any_statistics(
    builder: Callable[..., InitializedModel],
    input_norm: bool,
) -> None:
    """Verify declared periods hold in physical units whatever the normalization statistics are.

    Args:
        builder: Built-in model factory under test.
        input_norm: Whether the model normalizes its inputs.
    """
    period = 0.84
    model = builder(
        jax.random.key(0),
        2,
        1,
        input_mean=[0.5, 1.59],
        input_std=[0.4, 0.93],
        input_periods={1: period},
        input_norm=input_norm,
        periodic_features=True,
    )
    inputs = jnp.asarray([[0.7, 0.25], [0.1, 2.3]], dtype=jnp.float32)

    def gap(shift: float) -> float:
        """Return the largest output change after shifting the periodic coordinate.

        Args:
            shift: Shift applied to the periodic coordinate in physical units.

        Returns:
            Maximum absolute output difference.
        """
        shifted = inputs.at[:, 1].add(shift)
        return float(jnp.max(jnp.abs(model.apply(model.state, inputs) - model.apply(model.state, shifted))))

    assert model.input_periods == ((1, period),)
    assert gap(period) < 1e-5
    assert gap(1.0) > 1e-3


@pytest.mark.parametrize("builder", [build_mlp, build_modified_mlp, build_pirate_net])
def test_model_factories_merge_declared_and_configured_periods(builder: Callable[..., InitializedModel]) -> None:
    """Verify configured periods override or add to declared periods, and disabled features report none.

    Args:
        builder: Built-in model factory under test.
    """

    def periods(input_periods: dict[int, float] | None, configured: object = None, enabled: bool = True) -> object:
        """Build one model and return the periods it reports.

        Args:
            input_periods: Periods declared by the DataModule or a checkpoint.
            configured: Optional `periodic_features_kwargs`.
            enabled: Whether periodic features are enabled.

        Returns:
            Resolved periods reported by the factory.
        """
        model = builder(
            jax.random.key(0),
            2,
            1,
            input_periods=input_periods,
            periodic_features=enabled,
            periodic_features_kwargs=configured,
        )
        return model.input_periods

    assert periods({1: 0.8}) == ((1, 0.8),)
    assert periods({1: 0.8}, {"periods": {1: 0.5}}) == ((1, 0.5),)
    assert periods({1: 0.8}, {"periods": {0: 3.0}}) == ((0, 3.0), (1, 0.8))
    assert periods(None, OmegaConf.create({"periods": {"1": 0.5}})) == ((1, 0.5),)
    assert periods({1: 0.8}, enabled=False) is None
    with pytest.raises(ValueError, match="requires periods"):
        periods(None)
