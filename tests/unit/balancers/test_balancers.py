from typing import cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.balancers import ExactNTKBalancer, StaticLossBalancer, exact_ntk_trace
from phijax.core import BasePhiModule


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


@pytest.mark.parametrize("value", [float("nan"), float("inf")], ids=["nan", "inf"])
def test_ntk_non_finite_traces_preserve_previous_weights(value: float) -> None:
    """Verify one NaN or infinite mean diagonal leaves every weight unchanged and stays visible as a diagnostic.

    Args:
        value: Non-finite mean diagonal of the second loss.
    """
    balancer = ExactNTKBalancer(("a", "b"), update_every_n_steps=1, kernel_size=1, initial_weights={"a": 2.0, "b": 3.0})
    state = balancer.update_from_traces(jnp.asarray([1.0, value]), balancer.initialize())
    np.testing.assert_array_equal(state.weights, [2.0, 3.0])
    np.testing.assert_array_equal(state.traces, [1.0, value])


def test_ntk_update_averages_pointwise_diagonals_independently_of_batch_size() -> None:
    """Verify duplicated kernel samples do not change the mean diagonal-NTK diagnostic."""
    balancer = ExactNTKBalancer(("linear",), update_every_n_steps=1, kernel_size=1, moving_average_coefficient=0.0)
    update = balancer.make_update(cast(BasePhiModule, _LinearResidualModule()), kernel_chunk_size=2)
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
    module = cast(BasePhiModule, _LinearResidualModule())
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
        balancer.make_update(cast(BasePhiModule, _LinearResidualModule()), kernel_chunk_size=kernel_chunk_size)


def test_exact_ntk_update_plan_declares_fixed_diagnostic_batches() -> None:
    """Verify generic adaptive assembly samples exact-NTK batches without concrete type checks."""
    balancer = ExactNTKBalancer(
        ("linear",),
        update_every_n_steps=25,
        kernel_size=3,
        kernel_chunk_size=1,
        update_start_step=5,
    )
    plan = balancer.build_update_plan(cast(BasePhiModule, _LinearResidualModule()), ("data", "pde"))

    assert plan.batch_sizes == {"data": 3, "pde": 3}
    assert plan.every_n_steps == 25
    assert plan.update_start_step == 5
    assert callable(plan.update)


@pytest.mark.parametrize("resample", [False, True])
def test_exact_ntk_update_plan_sets_kernel_size_per_batch_key(resample: bool) -> None:
    """Verify per-key kernel sizes, full-pool `"all"` sizes, and `resample` reach the update plan.

    Args:
        resample: Whether diagnostic batches are redrawn for every update.
    """
    module = cast(BasePhiModule, _LinearResidualModule())
    balancer = ExactNTKBalancer(
        ("linear",),
        update_every_n_steps=1,
        kernel_size={"pde": 16, "boundary": "all"},
        resample=resample,
    )

    plan = balancer.build_update_plan(module, ("boundary", "pde"))
    shared = ExactNTKBalancer(("linear",), update_every_n_steps=1, kernel_size="all").build_update_plan(
        module, ("boundary", "pde")
    )

    assert plan.batch_sizes == {"boundary": "all", "pde": 16}
    assert plan.resample is resample
    assert shared.batch_sizes == {"boundary": "all", "pde": "all"}
    with pytest.raises(ValueError, match=r"missing \['initial'\], unknown \[\]"):
        balancer.build_update_plan(module, ("boundary", "initial", "pde"))
    with pytest.raises(ValueError, match=r"missing \[\], unknown \['pde'\]"):
        balancer.build_update_plan(module, ("boundary",))


def test_exact_ntk_defaults_update_start_to_one_interval() -> None:
    """Verify an omitted start step delays the first NTK update by one interval."""
    balancer = ExactNTKBalancer(("linear",), update_every_n_steps=25, kernel_size=3)

    plan = balancer.build_update_plan(cast(BasePhiModule, _LinearResidualModule()), ("data", "pde"))

    assert balancer.update_start_step == 25
    assert plan.update_start_step == 25


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"update_every_n_steps": True, "kernel_size": 2}, "update_every_n_steps"),
        ({"update_every_n_steps": 1, "update_start_step": True, "kernel_size": 2}, "update_start_step"),
        ({"update_every_n_steps": 1, "kernel_size": 1.5}, "kernel_size"),
        ({"update_every_n_steps": 1, "kernel_size": "half"}, "kernel_size"),
        ({"update_every_n_steps": 1, "kernel_size": {"pde": 1.5}}, "kernel_size"),
        ({"update_every_n_steps": 1, "kernel_size": 2, "resample": 1}, "resample"),
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
        ({"update_every_n_steps": 1, "kernel_size": {"pde": 0}}, "kernel_size"),
        ({"update_every_n_steps": 1, "kernel_size": {}}, "kernel_size"),
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


class _TwoStreamModule:
    """Expose two linear residual streams that read differently sized batches."""

    def __init__(self, stream_keys: dict[str, tuple[str, ...]]) -> None:
        """Store the batch keys each stream reads.

        Args:
            stream_keys: Batch keys reported for each stream name.
        """
        self.stream_keys = stream_keys

    def stream_batch_keys(self, name: str) -> tuple[str, ...]:
        """Return the batch keys read by one stream.

        Args:
            name: Stream name.

        Returns:
            Configured batch keys.
        """
        return self.stream_keys[name]

    def residual_stream(
        self,
        name: str,
        model_state: dict[str, jax.Array],
        batches: dict[str, dict[str, jax.Array]],
    ) -> jax.Array:
        """Evaluate a two-feature residual `[w x, 2 w x]` on the stream's own batch.

        Args:
            name: Stream name, equal to the batch key it reads.
            model_state: Scalar linear weight mapping.
            batches: Named input batch mapping.

        Returns:
            Residual matrix whose pointwise diagonal NTK is `5 x**2`.
        """
        prediction = batches[name]["inputs"] * model_state["weight"]
        return jnp.concatenate((prediction, 2.0 * prediction), axis=-1)


@pytest.mark.parametrize("kernel_chunk_size", [1, None])
@pytest.mark.parametrize(("long_key", "short_key"), [("a", "b"), ("b", "a")])
def test_ntk_update_maps_each_stream_over_its_own_batch(
    kernel_chunk_size: int | None, long_key: str, short_key: str
) -> None:
    """Verify streams whose batches differ in size each average over all of their own rows.

    The pointwise diagonal NTK of `[w x, 2 w x]` is `5 x**2`, so each mean is `5 * mean(x**2)` over its own batch.
    Mapping every stream over all batches together either failed or, when the shorter batch sorted first, silently kept
    only three rows of the longer one: `5 * mean([1, 4, 9]) ~= 23.3` instead of `37.5`.

    Args:
        kernel_chunk_size: Sequential or fully vectorized sample evaluation.
        long_key: Batch key of the four-row batch.
        short_key: Batch key of the three-row batch.
    """
    names = tuple(sorted((long_key, short_key)))
    balancer = ExactNTKBalancer(names, update_every_n_steps=1, kernel_size={long_key: 4, short_key: "all"})
    module = cast(BasePhiModule, _TwoStreamModule({long_key: (long_key,), short_key: (short_key,)}))
    batches = {
        long_key: {"inputs": jnp.asarray([[1.0], [2.0], [3.0], [4.0]])},
        short_key: {"inputs": jnp.asarray([[1.0], [1.0], [1.0]])},
    }

    state = balancer.update(
        module, {"weight": jnp.asarray(0.5)}, batches, balancer.initialize(), kernel_chunk_size=kernel_chunk_size
    )

    expected = {long_key: 37.5, short_key: 5.0}
    np.testing.assert_allclose(state.traces, [expected[name] for name in names], rtol=1e-6)


def test_ntk_update_rejects_a_stream_whose_batches_differ_in_size() -> None:
    """Verify one stream reading two batches with different row counts fails clearly instead of truncating rows."""
    balancer = ExactNTKBalancer(("a", "b"), update_every_n_steps=1, kernel_size={"a": 4, "b": "all"})
    module = cast(BasePhiModule, _TwoStreamModule({"a": ("a", "b"), "b": ("b",)}))
    batches = {"a": {"inputs": jnp.ones((4, 1))}, "b": {"inputs": jnp.ones((3, 1))}}

    with pytest.raises(ValueError, match=r"stream `a` reads batches \['a', 'b'\] with different row counts \[3, 4\]"):
        balancer.update(module, {"weight": jnp.asarray(0.5)}, batches, balancer.initialize())


def test_device_blocks_keep_each_device_rows_in_one_column() -> None:
    """Verify rows split over devices become `[rows_per_device, devices]`, so column `d` holds device `d`'s rows."""
    from phijax.balancers.ntk import _device_blocks

    batches = {"pde": {"inputs": jnp.arange(6.0)[:, None]}}

    blocks = _device_blocks(batches, devices=3)

    np.testing.assert_array_equal(blocks["pde"]["inputs"][..., 0], [[0.0, 2.0, 4.0], [1.0, 3.0, 5.0]])


def test_rows_per_device_uses_the_common_integer_kernel_size() -> None:
    """Verify per-device rows come from a shared integer `kernel_size`, and `all` or mixed sizes disable blocking."""
    mapped = ExactNTKBalancer(("a", "b"), update_every_n_steps=1, kernel_size={"pde": 8, "edge": "all", "wall": 4})
    uniform = ExactNTKBalancer(("a",), update_every_n_steps=1, kernel_size=16)

    assert uniform._rows_per_device(("pde",)) == 16
    assert mapped._rows_per_device(("pde",)) == 8
    assert mapped._rows_per_device(("edge",)) is None
    assert mapped._rows_per_device(("pde", "wall")) is None
