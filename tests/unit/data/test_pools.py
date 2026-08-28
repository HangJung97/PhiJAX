import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.data import (
    ChunkedPredictionSource,
    HostPool,
    NamedBatchSource,
    RandomRowSampler,
    SpaceTimeSampler,
    UniformDomainSampler,
    create_sampler,
)
from phijax.data.samplers import sample_pool


def _pool() -> HostPool:
    """Build a small deterministic host pool for immutable-storage and sampling tests.

    Returns:
        A four-row :class:`~phijax.data.pools.HostPool` with one auxiliary period field.
    """
    return HostPool(
        inputs=np.arange(12, dtype=np.float32).reshape(4, 3),
        targets=np.arange(4, dtype=np.float32).reshape(4, 1),
        aux={"period": np.ones((4, 1), dtype=np.float32)},
        metadata={"field_names": ("r", "th", "t")},
        reference_shape=(2, 2),
        flat_index=np.arange(4),
    )


def test_host_pool_copies_and_freezes_arrays() -> None:
    """Verify that pool storage is immutable and isolated from caller-owned arrays."""
    pool = _pool()
    assert not pool.inputs.flags.writeable
    assert not pool.targets.flags.writeable
    with pytest.raises(ValueError, match="read-only"):
        pool.inputs[0, 0] = -1.0


def test_host_pool_recursively_freezes_nested_metadata() -> None:
    """Verify application metadata mappings and their arrays cannot mutate after pool construction."""
    source_values = np.asarray([1.0, 2.0])
    pool = HostPool(
        inputs=np.ones((2, 1)),
        targets=np.ones((2, 1)),
        aux={},
        metadata={"artifact_metadata": {"scales": source_values}},
        reference_shape=(2,),
        flat_index=np.arange(2),
    )
    source_values[0] = -1.0
    nested = pool.metadata["artifact_metadata"]

    np.testing.assert_array_equal(nested["scales"], [1.0, 2.0])
    with pytest.raises(TypeError):
        nested["density"] = 1060.0
    with pytest.raises(ValueError, match="read-only"):
        nested["scales"][0] = -1.0


def test_device_sampling_is_reproducible_and_fixed_shape() -> None:
    """Verify that identical explicit keys produce identical fixed-shape batches."""
    pool = {name: jnp.asarray(value) for name, value in _pool().fields().items()}
    first = sample_pool(jax.random.key(2), pool, 7)
    second = sample_pool(jax.random.key(2), pool, 7)
    assert first["inputs"].shape == (7, 3)
    np.testing.assert_array_equal(first["inputs"], second["inputs"])


def test_prediction_chunks_are_deterministic_and_aligned() -> None:
    """Verify lazy prediction batches preserve fields, aligned padding, and validity masks."""
    host_pool = _pool()
    prediction_source = ChunkedPredictionSource(host_pool, 3)
    chunks = tuple(prediction_source)
    repeated = tuple(prediction_source)

    assert prediction_source.pool is host_pool
    assert len(prediction_source) == len(chunks) == 2
    assert set(chunks[0]) == {"inputs", "targets", "period", "mask"}
    assert chunks[0]["inputs"].shape == (3, 3)
    assert chunks[1]["inputs"].shape == (3, 3)
    assert isinstance(chunks[0]["inputs"], np.ndarray)
    assert isinstance(chunks[0]["mask"], np.ndarray)
    np.testing.assert_array_equal(chunks[0]["mask"], [True, True, True])
    np.testing.assert_array_equal(chunks[1]["mask"], [True, False, False])
    for batch, repeated_batch in zip(chunks, repeated, strict=True):
        for name in batch:
            np.testing.assert_array_equal(batch[name], repeated_batch[name])


def test_chunked_prediction_source_validates_batch_size_and_handles_empty_pools() -> None:
    """Verify lazy prediction chunks reject invalid sizes and preserve empty fixed shapes."""
    with pytest.raises(ValueError, match="positive integer"):
        ChunkedPredictionSource(_pool(), 0)
    empty_pool = HostPool(
        inputs=np.empty((0, 3), dtype=np.float32),
        targets=np.empty((0, 1), dtype=np.float32),
        aux={},
        metadata={},
        reference_shape=(0,),
        flat_index=np.empty((0,), dtype=np.int64),
    )
    chunks = tuple(ChunkedPredictionSource(empty_pool, 4))
    assert len(chunks) == 1
    assert chunks[0]["inputs"].shape == (4, 3)
    assert not bool(jnp.any(chunks[0]["mask"]))


def test_named_sampler_source_preserves_all_rows_and_step_determinism() -> None:
    """Verify finite `all` batches are exact while random batches derive from the global step."""
    sampler = RandomRowSampler(_pool().fields(), sort_axis=0)
    source = NamedBatchSource(
        {"complete": sampler, "random": sampler},
        {"complete": "all", "random": 3},
        jax.random.key(19),
    )

    first = source(4)
    repeated = source(4)

    np.testing.assert_array_equal(first["complete"]["inputs"], _pool().inputs)
    np.testing.assert_array_equal(first["random"]["inputs"], repeated["random"]["inputs"])
    assert first["random"]["inputs"].shape == (3, 3)


def test_named_sampler_source_prepares_persistent_state_on_selected_device() -> None:
    """Verify Trainer-facing preparation transfers sampler storage and keys exactly once per prepared source."""
    source = NamedBatchSource(
        {"random": RandomRowSampler(_pool().fields())},
        {"random": 2},
        jax.random.key(9),
    )
    sampler = source.samplers["random"]
    assert isinstance(sampler, RandomRowSampler)
    assert isinstance(sampler.pool["inputs"], np.ndarray)

    device = jax.devices("cpu")[0]
    prepared = source.prepare(device)
    prepared_sampler = prepared.samplers["random"]
    assert isinstance(prepared_sampler, RandomRowSampler)
    inputs = prepared_sampler.pool["inputs"]

    assert isinstance(inputs, jax.Array)
    assert inputs.devices() == {device}
    assert prepared.key.devices() == {device}
    np.testing.assert_array_equal(prepared(3)["random"]["inputs"], prepared(3)["random"]["inputs"])


def test_uniform_domain_sampler_uses_named_pool_bounds_and_constant_templates() -> None:
    """Verify `uniform_domain` generates fresh bounded coordinates and aligned templates."""
    pool = HostPool(
        inputs=np.asarray([[0.0, -2.0], [1.0, 2.0]], dtype=np.float32),
        targets=np.empty((2, 0), dtype=np.float32),
        aux={"coefficient": np.full((2, 1), 0.25, dtype=np.float32)},
        metadata={
            "coordinate_names": ("t", "x"),
            "sampling_bounds": (("t", 0.0, 1.0), ("x", -2.0, 2.0)),
        },
        reference_shape=(2,),
        flat_index=np.arange(2),
    )

    sampler = create_sampler(
        "uniform_domain",
        pool,
        options={"sort_axis": 0, "constant_fields": ["targets", "coefficient"]},
    )
    assert isinstance(sampler, UniformDomainSampler)
    batch = sampler.sample(jax.random.key(3), 32)

    assert batch["inputs"].shape == (32, 2)
    assert batch["targets"].shape == (32, 0)
    np.testing.assert_array_equal(batch["coefficient"], np.full((32, 1), 0.25))
    assert bool(jnp.all(jnp.diff(batch["inputs"][:, 0]) >= 0.0))
    assert bool(jnp.all((batch["inputs"][:, 1] >= -2.0) & (batch["inputs"][:, 1] <= 2.0)))


def test_space_time_sampler_replaces_time_and_preserves_spatial_rows() -> None:
    """Verify hybrid time-space sampling draws fresh time values over fixed spatial coordinates."""
    pool = HostPool(
        inputs=np.asarray([[0.0, -1.0], [0.0, 0.5], [0.0, 2.0]], dtype=np.float32),
        targets=np.empty((3, 0), dtype=np.float32),
        aux={},
        metadata={
            "coordinate_names": ("t", "x"),
            "sampling_bounds": (("t", 2.0, 3.0), ("x", -1.0, 2.0)),
        },
        reference_shape=(3,),
        flat_index=np.arange(3),
    )
    sampler = create_sampler("space_time", pool)
    assert isinstance(sampler, SpaceTimeSampler)

    batch = sampler.sample(jax.random.key(7), 12)

    assert bool(jnp.all((batch["inputs"][:, 0] >= 2.0) & (batch["inputs"][:, 0] <= 3.0)))
    assert set(np.asarray(batch["inputs"][:, 1])).issubset({-1.0, 0.5, 2.0})
    assert bool(jnp.all(jnp.diff(batch["inputs"][:, 0]) >= 0.0))


def test_sampler_names_and_generated_field_semantics_are_validated() -> None:
    """Verify short-name selection rejects unknown policies and varying generated templates."""
    with pytest.raises(ValueError, match="Unknown sampler name"):
        create_sampler("unknown", _pool())
    with pytest.raises(ValueError, match="varies by row"):
        create_sampler(
            "uniform_domain",
            _pool(),
            options={
                "bounds": [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
                "constant_fields": ["targets"],
            },
        )
