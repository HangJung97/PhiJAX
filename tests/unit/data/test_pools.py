import warnings
from typing import Any

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
    reconstruct_predictions,
)


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


def test_host_pool_supplies_safe_unsupervised_and_reconstruction_defaults() -> None:
    """Verify omitted pool fields produce immutable aligned defaults."""
    inputs = np.arange(6, dtype=np.float32).reshape(3, 2)

    pool = HostPool(inputs)

    assert pool.targets is not None
    assert pool.flat_index is not None
    assert pool.targets.shape == (3, 0)
    assert pool.targets.dtype == inputs.dtype
    assert pool.reference_shape == (3,)
    assert not pool.targets.flags.writeable
    np.testing.assert_array_equal(pool.flat_index, np.arange(3, dtype=np.int64))
    assert dict(pool.aux) == {}
    assert dict(pool.metadata) == {}


@pytest.mark.parametrize(
    ("aux", "match"),
    [
        ({"scale": np.asarray(2.0)}, "is 0-d"),
        ({"inputs": np.ones((2, 5))}, "cannot be named \\['inputs'\\]"),
        ({"targets": np.ones((2, 1))}, "cannot be named \\['targets'\\]"),
    ],
)
def test_host_pool_rejects_0d_and_reserved_auxiliary_arrays(aux: dict[str, np.ndarray], match: str) -> None:
    """Verify a 0-d aux array fails clearly and aux names cannot shadow the pool's own arrays.

    Args:
        aux: Invalid auxiliary fields.
        match: Expected error-message fragment.
    """
    with pytest.raises(ValueError, match=match):
        HostPool(inputs=np.zeros((2, 1), dtype=np.float32), aux=aux)


def test_host_pool_keeps_scalar_constants_out_of_sample_fields() -> None:
    """Verify Python and NumPy scalars stay in `aux` without becoming sample-wise fields."""
    pool = HostPool(inputs=np.zeros((2, 1), dtype=np.float32), aux={"scale": 2.0, "offset": np.float32(1.0)})

    assert set(pool.fields()) == {"inputs", "targets"}
    assert pool.aux["scale"] == 2.0


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


def _grid_inputs(*shape: int) -> np.ndarray:
    """Build dense integer-valued coordinates for a grid.

    Args:
        *shape: Grid axis lengths.

    Returns:
        Coordinates with shape `[*shape, len(shape)]`.
    """
    axes = [np.arange(size, dtype=np.float32) for size in shape]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1)


def test_host_pool_from_grid_flattens_a_full_grid_in_c_order() -> None:
    """Verify a full grid keeps every row in C order and records the grid for reconstruction."""
    inputs = _grid_inputs(2, 3)
    targets = inputs.sum(axis=-1, keepdims=True)
    pool = HostPool.from_grid(inputs, targets, aux={"weight": np.ones((2, 3)), "scale": 2.0}, metadata={"name": "g"})

    np.testing.assert_array_equal(pool.inputs, inputs.reshape(6, 2))
    np.testing.assert_array_equal(pool.targets, targets.reshape(6, 1))
    np.testing.assert_array_equal(pool.flat_index, np.arange(6))
    assert pool.aux["weight"].shape == (6,)
    assert pool.aux["scale"] == 2.0
    assert pool.metadata["name"] == "g"
    assert pool.reference_shape == (2, 3)
    np.testing.assert_array_equal(reconstruct_predictions(pool.targets, pool), targets)


@pytest.mark.parametrize("shape", [(3, 4), (1, 4), (3, 1, 2)], ids=["2d", "singleton-axis", "3d"])
def test_host_pool_from_grid_masks_rows_and_reconstructs_with_nan_outside(shape: tuple[int, ...]) -> None:
    """Verify a mask selects C-order rows, and reconstruction restores the grid with `NaN` outside the mask.

    Args:
        shape: Grid shape, including singleton axes.
    """
    inputs = _grid_inputs(*shape)
    mask = (np.arange(int(np.prod(shape))) % 2 == 0).reshape(shape)
    pool = HostPool.from_grid(inputs, inputs[..., :1], mask=mask)

    np.testing.assert_array_equal(pool.flat_index, np.flatnonzero(mask))
    np.testing.assert_array_equal(pool.inputs, inputs[mask])
    dense = reconstruct_predictions(pool.targets, pool)
    assert dense.shape == (*shape, 1)
    np.testing.assert_array_equal(dense[mask], inputs[mask][:, :1])
    assert bool(np.all(np.isnan(dense[~mask])))


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"inputs": np.zeros(3)}, "at least one grid axis"),
        ({"targets": np.zeros((3, 4))}, r"\[\*grid, output_features\]"),
        ({"targets": np.zeros((4, 3, 1))}, "Grid field `targets`"),
        ({"aux": {"weight": np.zeros((3, 5))}}, "Grid field `weight`"),
        ({"mask": np.ones((3, 4), dtype=np.int8)}, "Boolean array"),
        ({"mask": np.ones((4, 3), dtype=bool)}, "Boolean array"),
        ({"mask": np.zeros((3, 4), dtype=bool)}, "selects no rows"),
    ],
    ids=["rank-one", "targets-rank", "targets-shape", "aux-shape", "mask-dtype", "mask-shape", "empty-mask"],
)
def test_host_pool_from_grid_rejects_inconsistent_grids(kwargs: dict[str, Any], match: str) -> None:
    """Verify each malformed grid argument raises an error naming it.

    Args:
        kwargs: Arguments that replace the valid `[3, 4]` grid defaults.
        match: Expected error message pattern.
    """
    arguments = {"inputs": _grid_inputs(3, 4), **kwargs}
    with pytest.raises(ValueError, match=match):
        HostPool.from_grid(**arguments)


def test_reconstruct_predictions_rejects_duplicate_flat_indices() -> None:
    """Verify reconstruction refuses indices that would overwrite each other on the dense grid."""
    pool = HostPool(inputs=np.zeros((2, 1), dtype=np.float32), reference_shape=(2,), flat_index=np.asarray([0, 0]))

    with pytest.raises(ValueError, match="must be unique"):
        reconstruct_predictions(np.ones((2, 1), dtype=np.float32), pool)


def test_device_sampling_is_reproducible_and_fixed_shape() -> None:
    """Verify that identical explicit keys produce identical fixed-shape batches."""
    pool = {name: jnp.asarray(value) for name, value in _pool().fields().items()}
    first = RandomRowSampler(pool).sample(jax.random.key(2), 7)
    second = RandomRowSampler(pool).sample(jax.random.key(2), 7)
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
    assert set(chunks[0]) == {"inputs", "targets", "period", "valid_predict_mask"}
    assert chunks[0]["inputs"].shape == (3, 3)
    assert chunks[1]["inputs"].shape == (3, 3)
    assert isinstance(chunks[0]["inputs"], np.ndarray)
    assert isinstance(chunks[0]["valid_predict_mask"], np.ndarray)
    np.testing.assert_array_equal(chunks[0]["valid_predict_mask"], [True, True, True])
    np.testing.assert_array_equal(chunks[1]["valid_predict_mask"], [True, False, False])
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
    assert not bool(jnp.any(chunks[0]["valid_predict_mask"]))


def test_chunked_prediction_source_all_yields_one_chunk_on_any_device_count() -> None:
    """Verify `"all"` yields every row in one unpadded chunk, is not scaled by devices, and keeps empty pools valid."""
    host_pool = _pool()
    source = ChunkedPredictionSource(host_pool, "all")
    (chunk,) = tuple(source)

    assert len(source) == 1
    assert source.for_devices(4) is source
    np.testing.assert_array_equal(chunk["inputs"], host_pool.inputs)
    assert bool(np.all(chunk["valid_predict_mask"]))
    with pytest.raises(ValueError, match="positive integer or `all`"):
        ChunkedPredictionSource(host_pool, "every")

    empty_pool = HostPool(inputs=np.empty((0, 3), dtype=np.float32))
    (empty_chunk,) = tuple(ChunkedPredictionSource(empty_pool, "all"))
    assert empty_chunk["inputs"].shape == (1, 3)
    assert not bool(np.any(empty_chunk["valid_predict_mask"]))


def test_chunked_prediction_source_keeps_a_mask_field_and_rejects_the_reserved_name() -> None:
    """Verify a pool's own `mask` field is padded like other fields, while `valid_predict_mask` is reserved."""
    inputs = np.zeros((2, 1), dtype=np.float32)
    pool = HostPool(inputs=inputs, aux={"mask": np.asarray([False, True])})

    (batch,) = ChunkedPredictionSource(pool, 3)
    np.testing.assert_array_equal(batch["mask"], [False, True, False])
    np.testing.assert_array_equal(batch["valid_predict_mask"], [True, True, False])

    reserved = HostPool(inputs=inputs, aux={"valid_predict_mask": np.asarray([True, False])})
    with pytest.raises(ValueError, match="auxiliary array named `valid_predict_mask`"):
        ChunkedPredictionSource(reserved, 2)


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


def test_named_sampler_source_constructs_ordered_random_samplers_from_pools() -> None:
    """Verify the finite-pool shorthand preserves requested ordering and deterministic sampling."""
    pools = {"first": _pool(), "second": _pool()}
    source = NamedBatchSource.from_pools(
        pools,
        {"first": 2, "second": "all"},
        jax.random.key(8),
        names=("second", "first"),
        replace=False,
        sort_axes={"first": 0},
    )

    assert tuple(source.samplers) == ("second", "first")
    assert tuple(source.batch_sizes) == ("second", "first")
    np.testing.assert_array_equal(source(3)["second"]["inputs"], _pool().inputs)
    np.testing.assert_array_equal(source(3)["first"]["inputs"], source(3)["first"]["inputs"])
    with pytest.raises(KeyError, match="missing"):
        NamedBatchSource.from_pools(pools, {"missing": 1}, jax.random.key(0))
    with pytest.raises(ValueError, match="unique non-empty"):
        NamedBatchSource.from_pools(pools, {"first": 1}, jax.random.key(0), names=("first", "first"))


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
        metadata={"coordinate_names": ("t", "x"), "sampling_bounds": ((0.0, 1.0), (-2.0, 2.0))},
        reference_shape=(2,),
        flat_index=np.arange(2),
    )

    templates = {"targets": np.empty((0,), dtype=np.float32), "coefficient": pool.aux["coefficient"][0]}
    sampler = UniformDomainSampler(pool.metadata["sampling_bounds"], templates, sort_axis=0)
    batch = sampler.sample(jax.random.key(3), 32)

    assert batch["inputs"].shape == (32, 2)
    assert batch["targets"].shape == (32, 0)
    np.testing.assert_array_equal(batch["coefficient"], np.full((32, 1), 0.25))
    assert bool(jnp.all(jnp.diff(batch["inputs"][:, 0]) >= 0.0))
    assert bool(jnp.all((batch["inputs"][:, 1] >= -2.0) & (batch["inputs"][:, 1] <= 2.0)))


def test_uniform_domain_sampler_resolves_float64_requests_without_warnings() -> None:
    """Verify float64 pool inputs yield float32 coordinates without JAX truncation warnings when x64 is disabled."""
    sampler = UniformDomainSampler([[0.0, 1.0]], {}, dtype=np.float64)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        batch = sampler.prepare(jax.devices("cpu")[0]).sample(jax.random.key(0), 4)

    assert batch["inputs"].dtype == jnp.float32


def test_space_time_sampler_replaces_time_and_preserves_spatial_rows() -> None:
    """Verify hybrid time-space sampling draws fresh time values over fixed spatial coordinates."""
    pool = HostPool(
        inputs=np.asarray([[0.0, -1.0], [0.0, 0.5], [0.0, 2.0]], dtype=np.float32),
        targets=np.empty((3, 0), dtype=np.float32),
        aux={},
        metadata={"coordinate_names": ("t", "x")},
        reference_shape=(3,),
        flat_index=np.arange(3),
    )
    sampler = SpaceTimeSampler(pool.fields(), time_bounds=(2.0, 3.0))

    batch = sampler.sample(jax.random.key(7), 12)

    assert bool(jnp.all((batch["inputs"][:, 0] >= 2.0) & (batch["inputs"][:, 0] <= 3.0)))
    assert set(np.asarray(batch["inputs"][:, 1])).issubset({-1.0, 0.5, 2.0})
    assert bool(jnp.all(jnp.diff(batch["inputs"][:, 0]) >= 0.0))


def test_generated_samplers_validate_templates_and_targets() -> None:
    """Verify generated coordinates reject an `inputs` template and space-time sampling rejects time-bound targets."""
    with pytest.raises(ValueError, match="cannot define `inputs`"):
        UniformDomainSampler([[0.0, 1.0]], {"inputs": np.zeros(1)})
    with pytest.raises(ValueError, match="cannot reuse targets"):
        SpaceTimeSampler(_pool().fields(), time_bounds=(0.0, 1.0))


def test_batch_sources_scale_per_device_sizes() -> None:
    """Verify `for_devices` multiplies integer batch sizes and chunk sizes, keeps `all`, and is free for one device."""
    pool = _pool()
    source = NamedBatchSource.from_pools({"a": pool, "b": pool}, {"a": 2, "b": "all"}, jax.random.key(0))

    assert source.for_devices(1) is source
    assert dict(source.for_devices(3).batch_sizes) == {"a": 6, "b": "all"}

    chunks = ChunkedPredictionSource(pool, 1)
    assert chunks.for_devices(1) is chunks
    assert chunks.for_devices(2).batch_size == 2
    assert len(chunks.for_devices(2)) == 2
