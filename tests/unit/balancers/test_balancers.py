import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.balancers import ExactNTKBalancer, StaticLossBalancer, exact_ntk_trace


class _LinearResidualModule:
    """Expose a two-feature linear residual stream for exact-NTK tests."""

    def residual_stream(
        self,
        name: str,
        model_state: dict[str, jax.Array],
        batches: dict[str, dict[str, jax.Array]],
    ) -> jax.Array:
        """Evaluate a two-feature residual stream.

        Args:
            name: Expected loss name.
            model_state: Scalar linear weight mapping.
            batches: Named input batch mapping.

        Returns:
            Residual matrix whose second feature is twice its first feature.

        Raises:
            KeyError: If `name` is not the supported test loss.
        """
        if name != "linear":
            raise KeyError(name)
        prediction = batches["data"]["inputs"] * model_state["weight"]
        return jnp.concatenate((prediction, 2.0 * prediction), axis=-1)


def test_static_balancer_uses_stable_name_order() -> None:
    """Verify that static weights align with configured names rather than mapping insertion order."""
    balancer = StaticLossBalancer(("a", "b"), weights={"a": 2.0, "b": 3.0})
    state = balancer.initialize()
    total = balancer.combine({"b": jnp.asarray(5.0), "a": jnp.asarray(4.0)}, state)
    np.testing.assert_allclose(total, 23.0)
    assert set(balancer.diagnostics(state)) == {"weight/a", "weight/b"}


def test_exact_ntk_trace_matches_linear_jacobian_norm() -> None:
    """Verify that the exact trace equals the squared Frobenius norm of a known linear Jacobian."""
    inputs = jnp.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=jnp.float32)
    state = {"weight": jnp.asarray([0.5, -0.25], dtype=jnp.float32)}
    trace = jax.jit(lambda params: exact_ntk_trace(lambda values: inputs @ values["weight"], params))(state)
    np.testing.assert_allclose(trace, np.sum(np.asarray(inputs) ** 2))


def test_ntk_weights_use_mean_diagonal_ratio_and_smoothing() -> None:
    """Verify the smoothed mean-diagonal-over-component weighting rule."""
    balancer = ExactNTKBalancer(("a", "b"), update_every_n_steps=1, kernel_size=1, moving_average_coefficient=0.5)
    state = balancer.update_from_traces(jnp.asarray([2.0, 6.0]), balancer.initialize())
    np.testing.assert_allclose(state.traces, [2.0, 6.0])
    np.testing.assert_allclose(state.weights, [1.5, 5.0 / 6.0], rtol=1e-6)


def test_ntk_update_averages_pointwise_diagonals_independently_of_batch_size() -> None:
    """Verify duplicated kernel samples do not change the mean diagonal-NTK diagnostic."""
    balancer = ExactNTKBalancer(("linear",), update_every_n_steps=1, kernel_size=1, moving_average_coefficient=0.0)
    update = balancer.make_update(_LinearResidualModule(), kernel_chunk_size=2)
    model_state = {"weight": jnp.asarray(0.5, dtype=jnp.float32)}
    small_inputs = jnp.asarray([[1.0], [3.0]], dtype=jnp.float32)
    duplicated_inputs = jnp.concatenate((small_inputs, small_inputs), axis=0)

    small_state = update(model_state, {"data": {"inputs": small_inputs}}, balancer.initialize())
    duplicated_state = update(model_state, {"data": {"inputs": duplicated_inputs}}, balancer.initialize())

    # Per point, the two residual features contribute x**2 + (2*x)**2 = 5*x**2.
    np.testing.assert_allclose(small_state.traces, [25.0])
    np.testing.assert_allclose(duplicated_state.traces, small_state.traces)
    np.testing.assert_allclose(small_state.weights, [1.0])


def test_ntk_chunking_matches_sequential_and_fully_vectorized_computation() -> None:
    """Verify chunking and its remainder preserve sequential and full-vectorization results."""
    balancer = ExactNTKBalancer(("linear",), update_every_n_steps=1, kernel_size=1, moving_average_coefficient=0.0)
    module = _LinearResidualModule()
    model_state = {"weight": jnp.asarray(0.5, dtype=jnp.float32)}
    batches = {"data": {"inputs": jnp.arange(1.0, 6.0, dtype=jnp.float32).reshape(-1, 1)}}

    sequential = balancer.make_update(module, kernel_chunk_size=1)(model_state, batches, balancer.initialize())
    chunked = balancer.make_update(module, kernel_chunk_size=2)(model_state, batches, balancer.initialize())
    vectorized = balancer.make_update(module, kernel_chunk_size=None)(model_state, batches, balancer.initialize())

    np.testing.assert_allclose(chunked.traces, sequential.traces)
    np.testing.assert_allclose(vectorized.traces, sequential.traces)
    np.testing.assert_allclose(sequential.traces, [55.0])


@pytest.mark.parametrize("kernel_chunk_size", [True, 0, -1])
def test_ntk_update_rejects_invalid_kernel_chunk_sizes(kernel_chunk_size: int) -> None:
    """Verify memory-policy chunk sizes must be positive integers.

    Args:
        kernel_chunk_size: Invalid chunk size supplied by the parameterized test case.
    """
    balancer = ExactNTKBalancer(("linear",), update_every_n_steps=1, kernel_size=1)
    with pytest.raises(ValueError, match="kernel_chunk_size"):
        balancer.make_update(_LinearResidualModule(), kernel_chunk_size=kernel_chunk_size)


def test_exact_ntk_update_plan_declares_fixed_diagnostic_batches() -> None:
    """Verify generic adaptive assembly samples exact-NTK batches without concrete type checks."""
    balancer = ExactNTKBalancer(
        ("linear",),
        update_every_n_steps=25,
        kernel_size=3,
        kernel_chunk_size=1,
        update_start_step=5,
    )
    plan = balancer.build_update_plan(_LinearResidualModule(), ("data", "pde"))

    assert plan.batch_sizes == {"data": 3, "pde": 3}
    assert plan.every_n_steps == 25
    assert plan.update_start_step == 5
    assert callable(plan.update)


def test_exact_ntk_defaults_update_start_to_one_interval() -> None:
    """Verify an omitted start step delays the first NTK update by one interval."""
    balancer = ExactNTKBalancer(("linear",), update_every_n_steps=25, kernel_size=3)

    plan = balancer.build_update_plan(_LinearResidualModule(), ("data", "pde"))

    assert balancer.update_start_step == 25
    assert plan.update_start_step == 25


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"update_every_n_steps": True, "kernel_size": 2}, "update_every_n_steps"),
        ({"update_every_n_steps": 1, "update_start_step": True, "kernel_size": 2}, "update_start_step"),
        ({"update_every_n_steps": 1, "kernel_size": 1.5}, "kernel_size"),
        ({"update_every_n_steps": 1, "kernel_size": 2, "kernel_chunk_size": 1.5}, "kernel_chunk_size"),
    ],
)
def test_exact_ntk_rejects_non_integer_update_settings(kwargs: dict[str, object], message: str) -> None:
    """Verify adaptive scheduling and diagnostic sizes cannot be silently truncated.

    Args:
        kwargs: Invalid constructor settings.
        message: Expected configuration name in the exception message.
    """
    with pytest.raises(TypeError, match=message):
        ExactNTKBalancer(("linear",), **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"update_every_n_steps": 0, "kernel_size": 2}, "update_every_n_steps"),
        ({"update_every_n_steps": 1, "update_start_step": -1, "kernel_size": 2}, "update_start_step"),
        ({"update_every_n_steps": 1, "kernel_size": 0}, "kernel_size"),
        ({"update_every_n_steps": 1, "kernel_size": 2, "kernel_chunk_size": 0}, "kernel_chunk_size"),
    ],
)
def test_exact_ntk_rejects_non_positive_update_settings(kwargs: dict[str, object], message: str) -> None:
    """Verify adaptive scheduling and diagnostic sizes must be positive.

    Args:
        kwargs: Invalid constructor settings.
        message: Expected configuration name in the exception message.
    """
    with pytest.raises(ValueError, match=message):
        ExactNTKBalancer(("linear",), **kwargs)  # type: ignore[arg-type]
