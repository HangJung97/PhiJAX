import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest

from phijax.balancers import BalancerState
from phijax.training import initialize_train_state


def test_initialize_train_state_uses_stable_independent_prng_streams() -> None:
    """Verify runtime, sampling, and balancing keys follow the documented split order."""
    root_key = jax.random.key(9)
    expected_runtime, expected_sampling, expected_balancer = jax.random.split(root_key, 3)

    state = initialize_train_state(
        {"weight": jnp.asarray(1.0)},
        optax.sgd(1.0e-2),
        BalancerState(weights=jnp.ones(1), traces=jnp.zeros(1)),
        root_key,
    )

    np.testing.assert_array_equal(jax.random.key_data(state.rng_key), jax.random.key_data(expected_runtime))
    np.testing.assert_array_equal(jax.random.key_data(state.sampling_key), jax.random.key_data(expected_sampling))
    np.testing.assert_array_equal(jax.random.key_data(state.balancer_key), jax.random.key_data(expected_balancer))
    assert not bool(jnp.array_equal(jax.random.key_data(state.sampling_key), jax.random.key_data(state.balancer_key)))


def test_initialize_train_state_preserves_explicit_persistent_keys() -> None:
    """Verify supplied sampling and balancer keys are stored without another split."""
    runtime_key, sampling_key, balancer_key = jax.random.split(jax.random.key(4), 3)

    state = initialize_train_state(
        {"weight": jnp.asarray(1.0)},
        optax.sgd(1.0e-2),
        BalancerState(weights=jnp.ones(1), traces=jnp.zeros(1)),
        runtime_key,
        sampling_key=sampling_key,
        balancer_key=balancer_key,
    )

    np.testing.assert_array_equal(jax.random.key_data(state.rng_key), jax.random.key_data(runtime_key))
    np.testing.assert_array_equal(jax.random.key_data(state.sampling_key), jax.random.key_data(sampling_key))
    np.testing.assert_array_equal(jax.random.key_data(state.balancer_key), jax.random.key_data(balancer_key))


@pytest.mark.parametrize("missing_key", ["sampling", "balancer"])
def test_initialize_train_state_rejects_one_missing_persistent_key(missing_key: str) -> None:
    """Verify sampling and balancer keys must be supplied together.

    Args:
        missing_key: Persistent key omitted by the test case.
    """
    sampling_key = None if missing_key == "sampling" else jax.random.key(1)
    balancer_key = None if missing_key == "balancer" else jax.random.key(2)

    with pytest.raises(ValueError, match="both be supplied"):
        initialize_train_state(
            {"weight": jnp.asarray(1.0)},
            optax.sgd(1.0e-2),
            BalancerState(weights=jnp.ones(1), traces=jnp.zeros(1)),
            jax.random.key(0),
            sampling_key=sampling_key,
            balancer_key=balancer_key,
        )
