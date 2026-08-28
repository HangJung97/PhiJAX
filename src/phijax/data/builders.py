from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from phijax.data.io import ArrayFormat, get_array, load_arrays
from phijax.data.pools import HostPool


@dataclass(frozen=True)
class _Coordinate:
    """Store one resolved coordinate vector and its optional sampling bounds.

    Attributes:
        values: Strictly increasing one-dimensional coordinate values.
        bounds: Explicit or data-derived lower and upper sampling bounds. A singleton coordinate without explicit
            bounds uses `None` because it cannot define a nonzero sampling interval.
    """

    values: NDArray[Any]
    bounds: tuple[float, float] | None


@dataclass(frozen=True)
class _Field:
    """Store one resolved feature field on named coordinate axes.

    Attributes:
        values: Array with sample axes followed by one feature axis.
        axes: Coordinate names corresponding to the sample axes of `values`.
    """

    values: NDArray[Any]
    axes: tuple[str, ...]


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    """Validate one declarative-builder mapping.

    Args:
        value: Candidate mapping.
        name: Configuration path used in validation errors.

    Returns:
        Validated mapping.

    Raises:
        TypeError: If `value` is not a mapping.
    """
    if not isinstance(value, Mapping):
        raise TypeError(f"`{name}` must be a mapping.")
    return value


def _require_names(value: Any, *, name: str, allow_empty: bool = True) -> tuple[str, ...]:
    """Validate an ordered sequence of unique field names.

    Args:
        value: Candidate name sequence.
        name: Configuration path used in validation errors.
        allow_empty: Whether an empty sequence is valid.

    Returns:
        Ordered tuple of unique non-empty names.

    Raises:
        TypeError: If `value` is not a non-string sequence.
        ValueError: If a name is invalid, duplicated, or unexpectedly absent.
    """
    if not isinstance(value, Sequence) or isinstance(value, str):
        raise TypeError(f"`{name}` must be a sequence of names.")
    names = tuple(value)
    if any(not isinstance(item, str) or not item.strip() for item in names):
        raise ValueError(f"`{name}` must contain non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"`{name}` must not contain duplicate names.")
    if not allow_empty and not names:
        raise ValueError(f"`{name}` must not be empty.")
    return names


def _resolve_bounds(value: Any, *, name: str) -> tuple[float, float]:
    """Validate explicit lower and upper coordinate bounds.

    Args:
        value: Two-item bound sequence.
        name: Coordinate name used in validation errors.

    Returns:
        Finite strictly increasing bounds.

    Raises:
        ValueError: If `value` does not contain two finite increasing numbers.
    """
    if not isinstance(value, Sequence) or isinstance(value, str) or len(value) != 2:
        raise ValueError(f"Coordinate `{name}` bounds must contain two values.")
    bounds = (float(value[0]), float(value[1]))
    if not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
        raise ValueError(f"Coordinate `{name}` bounds must be finite and strictly increasing.")
    return bounds


def _resolve_coordinates(
    arrays: Mapping[str, Any],
    specifications: Mapping[str, Any],
    *,
    dtype: np.dtype[Any],
) -> dict[str, _Coordinate]:
    """Resolve coordinate specifications against a loaded array tree.

    Args:
        arrays: Loaded source arrays.
        specifications: Coordinate-name to source specification mapping.
        dtype: Floating NumPy dtype used by constructed pools.

    Returns:
        Resolved coordinates in configuration order.

    Raises:
        ValueError: If no coordinates are defined or a coordinate is invalid.
    """
    if not specifications:
        raise ValueError("`coordinates` must define at least one coordinate.")
    resolved: dict[str, _Coordinate] = {}
    for name, specification in specifications.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Coordinate names must be non-empty strings.")
        if isinstance(specification, str):
            key = specification
            configured_bounds = None
        else:
            spec = _require_mapping(specification, name=f"coordinates.{name}")
            key = spec.get("key", name)
            configured_bounds = spec.get("bounds")
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Coordinate `{name}` must use a non-empty source `key`.")
        values = np.asarray(get_array(arrays, key), dtype=dtype).squeeze()
        if values.ndim != 1 or values.size == 0:
            raise ValueError(f"Coordinate `{name}` must resolve to a non-empty one-dimensional array.")
        if not np.isfinite(values).all():
            raise ValueError(f"Coordinate `{name}` must contain only finite values.")
        if values.size > 1 and np.any(np.diff(values) <= 0.0):
            raise ValueError(f"Coordinate `{name}` must be strictly increasing.")
        if configured_bounds is not None:
            bounds = _resolve_bounds(configured_bounds, name=name)
        elif values.size > 1:
            bounds = (float(values[0]), float(values[-1]))
        else:
            bounds = None
        resolved[name] = _Coordinate(values=values, bounds=bounds)
    return resolved


def _resolve_constant_field(
    arrays: Mapping[str, Any],
    name: str,
    specification: Mapping[str, Any],
    *,
    dtype: np.dtype[Any],
) -> _Field:
    """Resolve one inline or keyed value that remains constant across sample axes.

    Args:
        arrays: Loaded source arrays.
        name: Configured field name.
        specification: Constant specification containing exactly one of `key` and `value`.
        dtype: Floating NumPy dtype used by constructed pools.

    Returns:
        Axis-independent field containing one scalar or fixed feature vector.

    Raises:
        ValueError: If source selection, axes, shape, or values are invalid.
    """
    axes = _require_names(specification.get("axes", ()), name=f"fields.{name}.axes")
    if axes:
        raise ValueError(f"Constant field `{name}` cannot declare sample axes.")
    has_key = "key" in specification
    has_value = "value" in specification
    if has_key == has_value:
        raise ValueError(f"Constant field `{name}` must define exactly one of `key` and `value`.")
    if has_key:
        key = specification["key"]
        if not isinstance(key, str) or not key.strip():
            raise ValueError(f"Constant field `{name}` must use a non-empty source `key`.")
        raw_value = get_array(arrays, key)
    else:
        raw_value = specification["value"]
    values = np.asarray(raw_value, dtype=dtype).squeeze()
    if values.ndim == 0:
        values = values.reshape(1)
    if values.ndim != 1 or values.size == 0:
        raise ValueError(f"Constant field `{name}` must resolve to a scalar or non-empty feature vector.")
    if not np.isfinite(values).all():
        raise ValueError(f"Constant field `{name}` must contain only finite values.")
    return _Field(values=values, axes=())


def _resolve_source_field(
    arrays: Mapping[str, Any],
    name: str,
    specification: Mapping[str, Any],
    coordinates: Mapping[str, _Coordinate],
    *,
    dtype: np.dtype[Any],
) -> _Field:
    """Resolve one source-backed scalar or vector field onto named coordinate axes.

    Args:
        arrays: Loaded source arrays.
        name: Configured field name.
        specification: Source key, axes, and optional feature-axis specification.
        coordinates: Previously resolved coordinates.
        dtype: Floating NumPy dtype used by constructed pools.

    Returns:
        Source field with a canonical trailing feature axis.

    Raises:
        ValueError: If axes, shapes, feature placement, or values are invalid.
    """
    if "value" in specification:
        raise ValueError(f"Source field `{name}` cannot define an inline `value`.")
    spec = specification
    key = spec.get("key", name)
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"Field `{name}` must use a non-empty source `key`.")
    axes = _require_names(spec.get("axes", ()), name=f"fields.{name}.axes")
    unknown_axes = tuple(axis for axis in axes if axis not in coordinates)
    if unknown_axes:
        raise ValueError(f"Field `{name}` references unknown coordinate axes: {unknown_axes}.")

    values = np.asarray(get_array(arrays, key), dtype=dtype)
    expected_shape = tuple(coordinates[axis].values.size for axis in axes)
    feature_axis = spec.get("feature_axis")
    if feature_axis is not None:
        if isinstance(feature_axis, bool) or not isinstance(feature_axis, int):
            raise ValueError(f"Field `{name}` feature axis must be an integer.")
        if values.ndim != len(axes) + 1:
            raise ValueError(f"Field `{name}` must have one feature axis in addition to its named coordinate axes.")
        values = np.moveaxis(values, feature_axis, -1)
    elif values.shape == expected_shape:
        values = values[..., None]
    elif values.ndim == len(axes) + 1 and values.shape[:-1] == expected_shape:
        pass
    elif values.size == int(np.prod(expected_shape, dtype=np.int64)):
        # SciPy may squeeze singleton MATLAB dimensions, which can be restored from the named coordinate sizes.
        values = values.reshape(expected_shape)[..., None]
    else:
        raise ValueError(
            f"Field `{name}` with axes {axes} must have sample shape {expected_shape}; received {values.shape}."
        )
    if values.shape[:-1] != expected_shape or values.shape[-1] < 1:
        raise ValueError(
            f"Field `{name}` with axes {axes} must have sample shape {expected_shape}; received {values.shape[:-1]}."
        )
    if not np.isfinite(values).all():
        raise ValueError(f"Field `{name}` must contain only finite values.")
    return _Field(values=values, axes=axes)


def _resolve_field(
    arrays: Mapping[str, Any],
    name: str,
    specification: Any,
    coordinates: Mapping[str, _Coordinate],
    *,
    dtype: np.dtype[Any],
) -> _Field:
    """Dispatch one declarative field specification by construction kind.

    Args:
        arrays: Loaded source arrays.
        name: Configured field name.
        specification: Field mapping with optional `kind`; `source` is the default.
        coordinates: Previously resolved coordinates.
        dtype: Floating NumPy dtype used by constructed pools.

    Returns:
        Canonical source-backed or constant field.

    Raises:
        ValueError: If the specification or field kind is unsupported.
    """
    if isinstance(specification, str):
        raise ValueError(f"Field `{name}` must declare its construction settings in a mapping.")
    spec = _require_mapping(specification, name=f"fields.{name}")
    kind = spec.get("kind", "source")
    if kind == "source":
        return _resolve_source_field(arrays, name, spec, coordinates, dtype=dtype)
    if kind == "constant":
        return _resolve_constant_field(arrays, name, spec, dtype=dtype)
    raise ValueError(f"Field `{name}` uses unsupported kind `{kind}`. Available kinds: constant, source.")


def _resolve_axis_slice(value: Any, *, axis: str, size: int) -> slice:
    """Resolve a scalar index or half-open index range without dropping an axis.

    Args:
        value: Integer index or two-item `[start, stop]` sequence. Range endpoints may be `None`.
        axis: Coordinate-axis name used in validation errors.
        size: Full coordinate size.

    Returns:
        Non-empty unit-step Python slice.

    Raises:
        ValueError: If the index or range is invalid or selects no coordinates.
    """
    if isinstance(value, Sequence) and not isinstance(value, str):
        if len(value) != 2:
            raise ValueError(f"Slice index range for `{axis}` must contain `[start, stop]`.")
        start, stop = value
        if any(item is not None and (isinstance(item, bool) or not isinstance(item, int)) for item in (start, stop)):
            raise ValueError(f"Slice index range for `{axis}` must contain integers or `None`.")
        resolved_start, resolved_stop, _ = slice(start, stop).indices(size)
    else:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"Slice index for `{axis}` must be an integer or `[start, stop]` range.")
        index = value if value >= 0 else size + value
        if index < 0 or index >= size:
            raise ValueError(f"Slice index {value} falls outside coordinate `{axis}` with size {size}.")
        resolved_start, resolved_stop = index, index + 1
    if resolved_start >= resolved_stop:
        raise ValueError(f"Slice for coordinate `{axis}` selects no values.")
    return slice(resolved_start, resolved_stop)


def _resolve_grid_slices(
    pool_name: str,
    axis_names: tuple[str, ...],
    coordinates: Mapping[str, _Coordinate],
    specification: Any,
) -> tuple[slice, ...]:
    """Resolve configured per-axis grid slices for one pool.

    Args:
        pool_name: Pool name used in validation errors.
        axis_names: Ordered pool input coordinates.
        coordinates: Resolved coordinate definitions.
        specification: Optional `slice` mapping from coordinate name to `{index: ...}`.

    Returns:
        One non-dropping slice per pool axis.

    Raises:
        ValueError: If a slice references an unknown axis or has an invalid specification.
    """
    if specification is None:
        return tuple(slice(None) for _ in axis_names)
    selections = _require_mapping(specification, name=f"pools.{pool_name}.slice")
    unknown_axes = tuple(axis for axis in selections if axis not in axis_names)
    if unknown_axes:
        raise ValueError(f"Pool `{pool_name}` slices unknown input coordinates: {unknown_axes}.")
    resolved: list[slice] = []
    for axis in axis_names:
        if axis not in selections:
            resolved.append(slice(None))
            continue
        selection = _require_mapping(selections[axis], name=f"pools.{pool_name}.slice.{axis}")
        if set(selection) != {"index"}:
            raise ValueError(f"Pool `{pool_name}` slice for `{axis}` must define only `index`.")
        resolved.append(_resolve_axis_slice(selection["index"], axis=axis, size=coordinates[axis].values.size))
    return tuple(resolved)


def _field_rows(
    field: _Field,
    *,
    pool_name: str,
    axis_names: tuple[str, ...],
    axis_slices: tuple[slice, ...],
    selected_shape: tuple[int, ...],
) -> NDArray[Any]:
    """Slice, align, broadcast, and flatten one field onto a pool grid.

    Args:
        field: Resolved source field.
        pool_name: Pool name used in validation errors.
        axis_names: Ordered pool coordinate axes.
        axis_slices: Per-axis slices applied by the pool.
        selected_shape: Shape of the selected pool grid.

    Returns:
        Rank-two rows with one column per field feature.

    Raises:
        ValueError: If the field depends on an axis absent from the pool inputs.
    """
    missing_axes = tuple(axis for axis in field.axes if axis not in axis_names)
    if missing_axes:
        raise ValueError(f"Pool `{pool_name}` cannot use a field defined on absent coordinate axes: {missing_axes}.")
    slices_by_axis = dict(zip(axis_names, axis_slices, strict=True))
    field_slices = tuple(slices_by_axis[axis] for axis in field.axes)
    values = field.values[(*field_slices, slice(None))]
    ordered_field_axes = tuple(axis for axis in axis_names if axis in field.axes)
    permutation = (*tuple(field.axes.index(axis) for axis in ordered_field_axes), len(field.axes))
    values = np.transpose(values, permutation)
    broadcast_shape = tuple(selected_shape[index] if axis in field.axes else 1 for index, axis in enumerate(axis_names))
    values = values.reshape(*broadcast_shape, values.shape[-1])
    return np.broadcast_to(values, (*selected_shape, values.shape[-1])).reshape(-1, values.shape[-1])


def _resolve_auxiliary_fields(value: Any, *, pool_name: str) -> dict[str, str]:
    """Resolve auxiliary output names to configured source-field names.

    Args:
        value: Sequence of field names or alias-to-field mapping.
        pool_name: Pool name used in validation errors.

    Returns:
        Ordered auxiliary alias-to-source mapping.

    Raises:
        TypeError: If the auxiliary specification is neither a mapping nor a sequence.
        ValueError: If an alias or source name is invalid.
    """
    if value is None:
        return {}
    if isinstance(value, Mapping):
        resolved = dict(value)
        if any(
            not isinstance(alias, str) or not alias.strip() or not isinstance(source, str) or not source.strip()
            for alias, source in resolved.items()
        ):
            raise ValueError(f"Pool `{pool_name}` auxiliary aliases and field names must be non-empty strings.")
        return resolved
    names = _require_names(value, name=f"pools.{pool_name}.aux")
    return {name: name for name in names}


def _grid_pool(
    name: str,
    specification: Mapping[str, Any],
    coordinates: Mapping[str, _Coordinate],
    fields: Mapping[str, _Field],
    *,
    dtype: np.dtype[Any],
) -> HostPool:
    """Construct one file-backed Cartesian-grid host pool.

    Args:
        name: Pool name.
        specification: Declarative grid-pool specification.
        coordinates: Resolved coordinates.
        fields: Resolved scalar and vector fields.
        dtype: Floating NumPy dtype used by constructed arrays.

    Returns:
        Immutable grid-backed host pool.

    Raises:
        ValueError: If coordinates, fields, grid mode, or slices are invalid.
    """
    axis_names = _require_names(specification.get("inputs", ()), name=f"pools.{name}.inputs", allow_empty=False)
    unknown_inputs = tuple(axis for axis in axis_names if axis not in coordinates)
    if unknown_inputs:
        raise ValueError(f"Pool `{name}` references unknown input coordinates: {unknown_inputs}.")
    if specification.get("grid", "full") != "full":
        raise ValueError(f"Pool `{name}` only supports `grid: full` for file-backed coordinates.")

    axis_slices = _resolve_grid_slices(name, axis_names, coordinates, specification.get("slice"))
    full_shape = tuple(coordinates[axis].values.size for axis in axis_names)
    selected_coordinates = [
        coordinates[axis].values[selection] for axis, selection in zip(axis_names, axis_slices, strict=True)
    ]
    selected_shape = tuple(values.size for values in selected_coordinates)
    coordinate_grids = np.meshgrid(*selected_coordinates, indexing="ij")
    inputs = np.column_stack([grid.reshape(-1) for grid in coordinate_grids]).astype(dtype, copy=False)

    target_names = _require_names(specification.get("targets", ()), name=f"pools.{name}.targets")
    missing_targets = tuple(field_name for field_name in target_names if field_name not in fields)
    if missing_targets:
        raise ValueError(f"Pool `{name}` references unknown target fields: {missing_targets}.")
    target_columns = [
        _field_rows(
            fields[field_name],
            pool_name=name,
            axis_names=axis_names,
            axis_slices=axis_slices,
            selected_shape=selected_shape,
        )
        for field_name in target_names
    ]
    targets = np.concatenate(target_columns, axis=1) if target_columns else np.zeros((inputs.shape[0], 0), dtype=dtype)

    auxiliary_sources = _resolve_auxiliary_fields(specification.get("aux"), pool_name=name)
    missing_auxiliary = tuple(field_name for field_name in auxiliary_sources.values() if field_name not in fields)
    if missing_auxiliary:
        raise ValueError(f"Pool `{name}` references unknown auxiliary fields: {missing_auxiliary}.")
    auxiliary = {
        alias: _field_rows(
            fields[field_name],
            pool_name=name,
            axis_names=axis_names,
            axis_slices=axis_slices,
            selected_shape=selected_shape,
        )
        for alias, field_name in auxiliary_sources.items()
    }

    dense_indices = np.arange(int(np.prod(full_shape)), dtype=np.int64).reshape(full_shape)
    flat_index = dense_indices[axis_slices].reshape(-1)
    metadata: dict[str, Any] = {"coordinate_names": axis_names, "output_names": target_names}
    if all(coordinates[axis].bounds is not None for axis in axis_names):
        metadata["sampling_bounds"] = tuple(
            (axis, *cast(tuple[float, float], coordinates[axis].bounds)) for axis in axis_names
        )
    return HostPool(
        inputs=inputs,
        targets=targets,
        aux=auxiliary,
        metadata=metadata,
        reference_shape=full_shape,
        flat_index=flat_index,
    )


def _uniform_pool(
    name: str,
    specification: Mapping[str, Any],
    coordinates: Mapping[str, _Coordinate],
    fields: Mapping[str, _Field],
    *,
    dtype: np.dtype[Any],
) -> HostPool:
    """Construct one uniformly sampled coordinate-box host pool.

    Args:
        name: Pool name.
        specification: Declarative sampled-pool specification.
        coordinates: Resolved coordinates supplying default bounds.
        fields: Resolved fields available for constant auxiliary values.
        dtype: Floating NumPy dtype used by constructed arrays.

    Returns:
        Immutable uniformly sampled host pool.

    Raises:
        ValueError: If sampling, bounds, auxiliary fields, or size settings are invalid.
    """
    axis_names = _require_names(specification.get("inputs", ()), name=f"pools.{name}.inputs", allow_empty=False)
    unknown_inputs = tuple(axis for axis in axis_names if axis not in coordinates)
    if unknown_inputs:
        raise ValueError(f"Pool `{name}` references unknown input coordinates: {unknown_inputs}.")
    if specification.get("slice") is not None:
        raise ValueError(f"Uniformly sampled pool `{name}` cannot also define `slice`.")
    if specification.get("targets"):
        raise ValueError(f"Uniformly sampled pool `{name}` cannot read grid-aligned target fields.")
    sampling = _require_mapping(specification.get("sampling"), name=f"pools.{name}.sampling")
    if sampling.get("method") != "uniform":
        raise ValueError(f"Pool `{name}` only supports `sampling.method: uniform`.")
    size = sampling.get("size")
    seed = sampling.get("seed", 0)
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"Pool `{name}` sampling size must be a positive integer.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError(f"Pool `{name}` sampling seed must be an integer.")
    missing_bounds = tuple(axis for axis in axis_names if coordinates[axis].bounds is None)
    if missing_bounds:
        raise ValueError(f"Pool `{name}` cannot infer nonzero sampling bounds for coordinates: {missing_bounds}.")

    generator = np.random.default_rng(seed)
    bounds = cast(tuple[tuple[float, float], ...], tuple(coordinates[axis].bounds for axis in axis_names))
    inputs = np.column_stack([generator.uniform(lower, upper, size) for lower, upper in bounds]).astype(dtype)
    auxiliary_sources = _resolve_auxiliary_fields(specification.get("aux"), pool_name=name)
    missing_auxiliary = tuple(field_name for field_name in auxiliary_sources.values() if field_name not in fields)
    if missing_auxiliary:
        raise ValueError(f"Pool `{name}` references unknown auxiliary fields: {missing_auxiliary}.")
    varying_auxiliary = tuple(field_name for field_name in auxiliary_sources.values() if fields[field_name].axes)
    if varying_auxiliary:
        raise ValueError(
            f"Uniformly sampled pool `{name}` only accepts constant auxiliary fields; received: {varying_auxiliary}."
        )
    auxiliary = {
        alias: np.broadcast_to(fields[field_name].values, (size, fields[field_name].values.size))
        for alias, field_name in auxiliary_sources.items()
    }
    return HostPool(
        inputs=inputs,
        targets=np.zeros((size, 0), dtype=dtype),
        aux=auxiliary,
        metadata={
            "coordinate_names": axis_names,
            "sampling_bounds": tuple((axis, *bound) for axis, bound in zip(axis_names, bounds, strict=True)),
        },
        reference_shape=(size,),
        flat_index=np.arange(size, dtype=np.int64),
    )


def _prepare_source(source: Mapping[str, Any]) -> tuple[Path, ArrayFormat]:
    """Resolve a source path and run an optional application preparation hook.

    Args:
        source: Source mapping with `path`, optional `file_format`, and optional callable `prepare`.

    Returns:
        Prepared path and requested array format.

    Raises:
        TypeError: If the path or preparation hook is invalid.
    """
    path = source.get("path")
    if not isinstance(path, (str, PathLike)):
        raise TypeError("`source.path` must be a string or path-like value.")
    prepare = source.get("prepare")
    if prepare is not None:
        if not isinstance(prepare, Callable):
            raise TypeError("`source.prepare` must be callable when provided.")
        prepared_path = prepare(path)
        if not isinstance(prepared_path, (str, PathLike)):
            raise TypeError("`source.prepare` must return a string or path-like value.")
        path = prepared_path
    file_format = source.get("file_format", "auto")
    return Path(path), cast(ArrayFormat, file_format)


def build_array_pools(
    *,
    source: Mapping[str, Any],
    coordinates: Mapping[str, Any],
    pools: Mapping[str, Any],
    fields: Mapping[str, Any] | None = None,
    dtype: str = "float32",
) -> dict[str, HostPool]:
    """Build named host pools declaratively from an NPZ, MATLAB, or HDF5 artifact.

    Coordinate arrays define reusable grid axes and, unless explicitly overridden, their uniform-sampling bounds.
    File-backed pools form Cartesian grids, optionally sliced by a scalar index or half-open `[start, stop]` range.
    Uniform pools sample directly from the bounds of their configured input coordinates. Losses remain independent and
    share a sampled pool by referencing the same objective `batch_key`.

    Args:
        source: Mapping with `path`, optional `file_format`, and optional callable `prepare` invoked before loading.
        coordinates: Named one-dimensional coordinate specifications. Each value is a source key or a mapping with
            `key` and optional explicit `bounds`.
        pools: Named grid or uniform-sampling pool specifications.
        fields: Optional named source or constant field specifications. Source fields contain `key`, `axes`, and an
            optional `feature_axis`; constant fields contain exactly one of `key` and `value` and no sample axes.
        dtype: Floating NumPy dtype used for coordinates, fields, and generated samples.

    Returns:
        Non-empty mapping of configured names to immutable :class:`HostPool` objects.

    Raises:
        TypeError: If a major configuration section has the wrong container type.
        ValueError: If the dtype, coordinates, fields, pool references, selections, or sampling settings are invalid.
    """
    resolved_dtype = np.dtype(dtype)
    if not np.issubdtype(resolved_dtype, np.floating):
        raise ValueError("Declarative array pools require a floating `dtype`.")
    source_mapping = _require_mapping(source, name="source")
    coordinate_mapping = _require_mapping(coordinates, name="coordinates")
    pool_mapping = _require_mapping(pools, name="pools")
    field_mapping = _require_mapping(fields or {}, name="fields")
    if not pool_mapping:
        raise ValueError("`pools` must define at least one named pool.")

    path, file_format = _prepare_source(source_mapping)
    arrays = load_arrays(path, file_format=file_format)
    resolved_coordinates = _resolve_coordinates(arrays, coordinate_mapping, dtype=resolved_dtype)
    resolved_fields = {
        name: _resolve_field(arrays, name, specification, resolved_coordinates, dtype=resolved_dtype)
        for name, specification in field_mapping.items()
    }

    result: dict[str, HostPool] = {}
    for name, specification in pool_mapping.items():
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Pool names must be non-empty strings.")
        spec = _require_mapping(specification, name=f"pools.{name}")
        if spec.get("sampling") is None:
            result[name] = _grid_pool(name, spec, resolved_coordinates, resolved_fields, dtype=resolved_dtype)
        else:
            result[name] = _uniform_pool(name, spec, resolved_coordinates, resolved_fields, dtype=resolved_dtype)
    return result
