from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
import numpy as np

from phijax.data.batching import DevicePool
from phijax.data.pools import HostPool
from phijax.types import JaxDevice

type SamplerName = Literal["random_rows", "uniform_domain", "space_time"]


class BatchSampler(ABC):
    """Define one immutable explicit-key batch sampling policy."""

    def prepare(self, device: JaxDevice) -> "BatchSampler":
        """Prepare persistent sampler state for one trainer-selected device.

        Stateless custom samplers may inherit this implementation. Samplers owning arrays should override it and
        return an immutable copy whose persistent storage resides on `device`.

        Args:
            device: Trainer Strategy's process-local root device.

        Returns:
            Device-ready sampler.
        """
        del device
        return self

    @abstractmethod
    def sample(self, key: jax.Array, batch_size: int) -> DevicePool:
        """Draw one fixed-size batch.

        Args:
            key: Explicit JAX PRNG key controlling this draw.
            batch_size: Positive number of requested rows.

        Returns:
            Sample-wise device arrays sharing the requested leading dimension.
        """

    def sample_all(self) -> DevicePool:
        """Return the complete finite source when supported.

        Returns:
            Complete sample-wise device arrays.

        Raises:
            ValueError: If the sampler represents a continuous distribution without a finite source.
        """
        raise ValueError(f"Sampler `{type(self).__name__}` does not support the `all` batch-size policy.")


@dataclass(frozen=True, slots=True)
class RandomRowSampler(BatchSampler):
    """Sample aligned rows from one finite pool.

    Attributes:
        pool: Sample-wise host or device arrays with a shared leading dimension.
        replace: Whether randomly drawn indices may repeat.
        sort_axis: Optional input-coordinate column used to sort every returned field consistently.
    """

    pool: Mapping[str, Any]
    replace: bool = True
    sort_axis: int | None = None

    def __post_init__(self) -> None:
        """Validate the finite source and optional input sort axis.

        Raises:
            KeyError: If the pool has no `inputs` field.
            ValueError: If pool fields have inconsistent row counts or `sort_axis` is invalid.
        """
        _validate_pool(self.pool)
        _validate_sort_axis(self.sort_axis, int(self.pool["inputs"].shape[1]))

    def prepare(self, device: JaxDevice) -> "RandomRowSampler":
        """Place the finite candidate pool once on the trainer-selected device.

        Args:
            device: Trainer Strategy's process-local root device.

        Returns:
            Immutable sampler sharing policy values with a device-resident pool.
        """
        if all(_is_on_device(value, device) for value in self.pool.values()):
            return self
        pool = {name: jax.device_put(value, device) for name, value in self.pool.items()}
        return replace(self, pool=pool)

    def sample(self, key: jax.Array, batch_size: int) -> DevicePool:
        """Sample aligned finite-pool rows.

        Args:
            key: Explicit JAX PRNG key controlling row indices.
            batch_size: Positive number of requested rows.

        Returns:
            Random aligned pool fields, optionally sorted by one input coordinate.
        """
        return _sort_batch(
            sample_pool(key, self.pool, batch_size, replace=self.replace),
            self.sort_axis,
        )

    def sample_all(self) -> DevicePool:
        """Return every finite-pool row exactly once.

        Returns:
            Complete aligned pool fields, optionally sorted by one input coordinate.
        """
        return _sort_batch({name: jnp.asarray(value) for name, value in self.pool.items()}, self.sort_axis)


@dataclass(frozen=True, slots=True)
class UniformDomainSampler(BatchSampler):
    """Generate fresh coordinates uniformly inside a rectangular domain.

    Attributes:
        bounds: Array with shape `[input_features, 2]` containing increasing lower and upper bounds.
        templates: Non-input field values broadcast to every generated coordinate.
        dtype: Floating dtype used for generated coordinates.
        sort_axis: Optional coordinate column used to sort each generated batch.
    """

    bounds: Any
    templates: Mapping[str, Any]
    dtype: jax.typing.DTypeLike = jnp.float32
    sort_axis: int | None = None

    def __post_init__(self) -> None:
        """Validate domain bounds, templates, dtype, and sort policy.

        Raises:
            ValueError: If bounds are not finite increasing intervals, a template is named `inputs`, or sorting is
                invalid.
        """
        bounds = np.asarray(self.bounds)
        if bounds.ndim != 2 or bounds.shape[1] != 2 or not np.isfinite(bounds).all():
            raise ValueError("Uniform-domain bounds must have shape `[input_features, 2]` and contain finite values.")
        if np.any(bounds[:, 0] >= bounds[:, 1]):
            raise ValueError("Every uniform-box lower bound must be smaller than its upper bound.")
        if "inputs" in self.templates:
            raise ValueError("Uniform-domain templates cannot define `inputs`.")
        if not np.issubdtype(np.dtype(self.dtype), np.floating):
            raise ValueError("Uniform-domain coordinates require a floating dtype.")
        _validate_sort_axis(self.sort_axis, bounds.shape[0])

    def prepare(self, device: JaxDevice) -> "UniformDomainSampler":
        """Place domain bounds and field templates once on the trainer-selected device.

        Args:
            device: Trainer Strategy's process-local root device.

        Returns:
            Immutable sampler with device-resident generation metadata.
        """
        if _is_on_device(self.bounds, device) and all(
            _is_on_device(value, device) for value in self.templates.values()
        ):
            return self
        templates = {name: jax.device_put(value, device) for name, value in self.templates.items()}
        return replace(self, bounds=jax.device_put(self.bounds, device), templates=templates)

    def sample(self, key: jax.Array, batch_size: int) -> DevicePool:
        """Generate one fresh fixed-size domain batch.

        Args:
            key: Explicit JAX PRNG key controlling coordinate generation.
            batch_size: Positive number of requested coordinates.

        Returns:
            Generated `inputs` and broadcast template fields.

        Raises:
            ValueError: If `batch_size` is not positive.
        """
        if batch_size < 1:
            raise ValueError("`batch_size` must be positive.")
        bounds = jnp.asarray(self.bounds, dtype=self.dtype)
        inputs = jax.random.uniform(
            key,
            shape=(batch_size, bounds.shape[0]),
            minval=bounds[:, 0],
            maxval=bounds[:, 1],
            dtype=self.dtype,
        )
        batch = {
            "inputs": inputs,
            **{name: jnp.broadcast_to(value, (batch_size, *value.shape)) for name, value in self.templates.items()},
        }
        return _sort_batch(batch, self.sort_axis)


@dataclass(frozen=True, slots=True)
class SpaceTimeSampler(BatchSampler):
    """Pair fresh uniform times with rows sampled from a fixed spatial mesh.

    Attributes:
        pool: Finite host or device pool whose `inputs` contain a replaceable temporal coordinate column.
        time_bounds: Increasing lower and upper time bounds.
        time_axis: Input-coordinate column replaced with fresh uniform times.
        replace: Whether sampled spatial rows may repeat.
        sort_axis: Optional input-coordinate column used to sort every returned field.
    """

    pool: Mapping[str, Any]
    time_bounds: tuple[float, float]
    time_axis: int = 0
    replace: bool = True
    sort_axis: int | None = 0

    def __post_init__(self) -> None:
        """Validate the finite mesh and temporal settings.

        Raises:
            ValueError: If bounds, the temporal axis, target semantics, or sorting are invalid.
        """
        _validate_pool(self.pool)
        bounds = np.asarray(self.time_bounds, dtype=np.float64)
        if bounds.shape != (2,) or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]:
            raise ValueError("`time_bounds` must contain two finite increasing values.")
        input_width = int(self.pool["inputs"].shape[1])
        if not 0 <= self.time_axis < input_width:
            raise ValueError("`time_axis` must select an input-coordinate column.")
        if "targets" in self.pool and int(self.pool["targets"].shape[1]) != 0:
            raise ValueError("Space-time sampling cannot reuse targets tied to the original time coordinates.")
        _validate_sort_axis(self.sort_axis, input_width)

    def prepare(self, device: JaxDevice) -> "SpaceTimeSampler":
        """Place the finite spatial mesh once on the trainer-selected device.

        Args:
            device: Trainer Strategy's process-local root device.

        Returns:
            Immutable sampler with device-resident mesh fields.
        """
        if all(_is_on_device(value, device) for value in self.pool.values()):
            return self
        pool = {name: jax.device_put(value, device) for name, value in self.pool.items()}
        return replace(self, pool=pool)

    def sample(self, key: jax.Array, batch_size: int) -> DevicePool:
        """Sample spatial rows and replace their temporal coordinate.

        Args:
            key: Explicit JAX PRNG key split across mesh and temporal draws.
            batch_size: Positive number of requested rows.

        Returns:
            Aligned pool fields with fresh uniformly generated time coordinates.
        """
        mesh_key, time_key = jax.random.split(key)
        batch = sample_pool(mesh_key, self.pool, batch_size, replace=self.replace)
        lower, upper = self.time_bounds
        times = jax.random.uniform(
            time_key,
            shape=(batch_size,),
            minval=jnp.asarray(lower, dtype=batch["inputs"].dtype),
            maxval=jnp.asarray(upper, dtype=batch["inputs"].dtype),
            dtype=batch["inputs"].dtype,
        )
        batch["inputs"] = batch["inputs"].at[:, self.time_axis].set(times)
        return _sort_batch(batch, self.sort_axis)


def create_sampler(
    name: SamplerName | str,
    pool: HostPool,
    *,
    options: Mapping[str, Any] | None = None,
) -> BatchSampler:
    """Create a built-in sampler from an explicit stable name.

    Sampler selection intentionally avoids Hydra `_target_` values. The surrounding DataModule resolves the short
    names `random_rows`, `uniform_domain`, and `space_time` while retaining typed constructors internally.

    Args:
        name: Built-in sampler name.
        pool: Immutable host pool supplying finite rows, field templates, or inferred coordinate bounds.
        options: Sampler-specific options such as `replace`, `sort_axis`, `bounds`, `constant_fields`, `time_bounds`,
            or `time_axis`.

    Returns:
        Configured immutable batch sampler.

    Raises:
        TypeError: If an option has an unsupported container or scalar type.
        ValueError: If the name, bounds, templates, or sampler options are invalid.
    """
    settings = dict(options or {})
    if name == "random_rows":
        replace = _pop_bool(settings, "replace", True)
        sort_axis = _pop_optional_int(settings, "sort_axis", None)
        _reject_options(name, settings)
        return RandomRowSampler(pool.fields(), replace=replace, sort_axis=sort_axis)
    if name == "uniform_domain":
        bounds = _resolve_pool_bounds(pool, settings.pop("bounds", None))
        sort_axis = _pop_optional_int(settings, "sort_axis", None)
        constant_fields = _pop_names(settings, "constant_fields")
        _reject_options(name, settings)
        templates = _constant_templates(pool, constant_fields)
        return UniformDomainSampler(
            bounds,
            templates,
            dtype=pool.inputs.dtype,
            sort_axis=sort_axis,
        )
    if name == "space_time":
        time_axis = _pop_optional_int(settings, "time_axis", 0)
        assert time_axis is not None
        time_bounds = _resolve_time_bounds(pool, settings.pop("time_bounds", None), time_axis)
        replace = _pop_bool(settings, "replace", True)
        sort_axis = _pop_optional_int(settings, "sort_axis", time_axis)
        _reject_options(name, settings)
        return SpaceTimeSampler(
            pool.fields(),
            time_bounds=time_bounds,
            time_axis=time_axis,
            replace=replace,
            sort_axis=sort_axis,
        )
    choices = ", ".join(("random_rows", "uniform_domain", "space_time"))
    raise ValueError(f"Unknown sampler name `{name}`. Available samplers: {choices}.")


def sample_pool(
    key: jax.Array,
    pool: Mapping[str, Any],
    batch_size: int,
    *,
    replace: bool = True,
) -> DevicePool:
    """Sample a fixed-shape device batch using an explicit PRNG key.

    Args:
        key: JAX PRNG key controlling index selection.
        pool: Host or device field mapping with a shared leading sample dimension.
        batch_size: Positive number of rows in the returned batch.
        replace: Whether indices may repeat.

    Returns:
        Field mapping with leading dimension `batch_size`.

    Raises:
        ValueError: If `batch_size` is not positive, the pool is empty, or sampling without replacement requests more
            rows than the pool contains.
        KeyError: If `pool` does not contain `inputs`.
    """
    if batch_size < 1:
        raise ValueError("`batch_size` must be positive.")
    row_count = int(pool["inputs"].shape[0])
    if row_count < 1:
        raise ValueError("Cannot sample an empty pool.")
    if not replace and batch_size > row_count:
        raise ValueError("Sampling without replacement cannot exceed the pool size.")
    indices = jax.random.choice(key, row_count, shape=(batch_size,), replace=replace)
    return {name: jnp.asarray(values)[indices] for name, values in pool.items()}


def _validate_pool(pool: Mapping[str, Any]) -> None:
    """Validate a host or device pool's sample dimensions.

    Args:
        pool: Candidate sample-wise array mapping.

    Raises:
        KeyError: If `inputs` is absent.
        ValueError: If inputs are not rank two, the pool is empty, or field row counts differ.
    """
    inputs = pool["inputs"]
    if inputs.ndim != 2:
        raise ValueError("Sampler pool `inputs` must be rank two.")
    row_count = int(inputs.shape[0])
    if row_count < 1:
        raise ValueError("Sampler pools cannot be empty.")
    if any(value.ndim < 1 or int(value.shape[0]) != row_count for value in pool.values()):
        raise ValueError("Every sampler pool field must share the `inputs` leading dimension.")


def _is_on_device(value: Any, device: JaxDevice) -> bool:
    """Report whether one value is a JAX array exclusively resident on the selected device.

    Args:
        value: Candidate persistent sampler leaf.
        device: Trainer-selected process-local root device.

    Returns:
        `True` when `value` is already placed exactly on `device`.
    """
    return isinstance(value, jax.Array) and value.devices() == {device}


def _validate_sort_axis(sort_axis: int | None, input_width: int) -> None:
    """Validate an optional coordinate sort column.

    Args:
        sort_axis: Optional input-coordinate column.
        input_width: Number of input-coordinate columns.

    Raises:
        ValueError: If `sort_axis` does not select an input column.
    """
    if sort_axis is not None and not 0 <= sort_axis < input_width:
        raise ValueError("`sort_axis` must select an input-coordinate column or be `None`.")


def _sort_batch(batch: DevicePool, sort_axis: int | None) -> DevicePool:
    """Sort all sample-wise fields by one input-coordinate column.

    Args:
        batch: Aligned sample-wise arrays.
        sort_axis: Optional input-coordinate column.

    Returns:
        Original-order batch when sorting is disabled, otherwise consistently reordered fields.
    """
    if sort_axis is None:
        return batch
    indices = jnp.argsort(batch["inputs"][:, sort_axis])
    return {name: values[indices] for name, values in batch.items()}


def _pop_bool(settings: dict[str, Any], name: str, default: bool) -> bool:
    """Remove and validate one Boolean sampler option.

    Args:
        settings: Mutable unresolved option mapping.
        name: Option name.
        default: Default used when the option is absent.

    Returns:
        Resolved Boolean option.

    Raises:
        TypeError: If the configured value is not Boolean.
    """
    value = settings.pop(name, default)
    if not isinstance(value, bool):
        raise TypeError(f"Sampler option `{name}` must be Boolean.")
    return value


def _pop_optional_int(settings: dict[str, Any], name: str, default: int | None) -> int | None:
    """Remove and validate one optional integer sampler option.

    Args:
        settings: Mutable unresolved option mapping.
        name: Option name.
        default: Default used when the option is absent.

    Returns:
        Resolved integer or `None`.

    Raises:
        TypeError: If the configured value is not an integer or `None`.
    """
    value = settings.pop(name, default)
    if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
        raise TypeError(f"Sampler option `{name}` must be an integer or `None`.")
    return value


def _pop_names(settings: dict[str, Any], name: str) -> tuple[str, ...]:
    """Remove and validate one sequence of unique field names.

    Args:
        settings: Mutable unresolved option mapping.
        name: Option name.

    Returns:
        Ordered unique non-empty field names, or an empty tuple when the option is absent.

    Raises:
        TypeError: If the configured value is not a non-string sequence.
        ValueError: If a field name is empty or repeated.
    """
    value = settings.pop(name, ())
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"Sampler option `{name}` must be a sequence of field names.")
    names = tuple(value)
    if any(not isinstance(field_name, str) or not field_name.strip() for field_name in names):
        raise ValueError(f"Sampler option `{name}` must contain non-empty strings.")
    if len(set(names)) != len(names):
        raise ValueError(f"Sampler option `{name}` must not contain duplicate field names.")
    return names


def _reject_options(name: str, settings: Mapping[str, Any]) -> None:
    """Reject unused sampler options to catch configuration mistakes.

    Args:
        name: Selected sampler name.
        settings: Remaining option mapping.

    Raises:
        ValueError: If unsupported options remain.
    """
    if settings:
        unknown = ", ".join(sorted(settings))
        raise ValueError(f"Sampler `{name}` received unsupported options: {unknown}.")


def _resolve_pool_bounds(pool: HostPool, configured: Any) -> np.ndarray:
    """Resolve uniform-box bounds explicitly or from pool metadata.

    Args:
        pool: Host pool supplying coordinate names and optional `sampling_bounds` metadata.
        configured: Optional mapping by coordinate name or array with shape `[input_features, 2]`.

    Returns:
        Floating bounds array in model-input column order.

    Raises:
        TypeError: If configured bounds use an unsupported container.
        ValueError: If bounds are absent, incomplete, or have the wrong shape.
    """
    coordinate_names = tuple(str(value) for value in pool.metadata.get("coordinate_names", ()))
    bounds = configured
    if bounds is None:
        metadata_bounds = pool.metadata.get("sampling_bounds")
        if metadata_bounds is not None:
            bounds = {str(axis): (lower, upper) for axis, lower, upper in metadata_bounds}
    if isinstance(bounds, Mapping):
        if len(coordinate_names) != pool.inputs.shape[1]:
            raise ValueError("Named bounds require pool `coordinate_names` metadata aligned with input columns.")
        missing = tuple(name for name in coordinate_names if name not in bounds)
        if missing:
            raise ValueError(f"Uniform-domain bounds are missing coordinates: {missing}.")
        values = np.asarray([bounds[name] for name in coordinate_names], dtype=pool.inputs.dtype)
    elif isinstance(bounds, Sequence) and not isinstance(bounds, (str, bytes)):
        values = np.asarray(bounds, dtype=pool.inputs.dtype)
    else:
        raise TypeError("Uniform-domain `bounds` must be a coordinate mapping or interval sequence.")
    if values.shape != (pool.inputs.shape[1], 2):
        raise ValueError("Uniform-domain bounds must align with every pool input column.")
    return values


def _resolve_time_bounds(pool: HostPool, configured: Any, time_axis: int) -> tuple[float, float]:
    """Resolve temporal bounds explicitly or from pool metadata.

    Args:
        pool: Host pool supplying optional coordinate and bound metadata.
        configured: Optional two-value temporal interval.
        time_axis: Input column interpreted as time.

    Returns:
        Increasing temporal interval.

    Raises:
        TypeError: If the interval is not a non-string sequence.
        ValueError: If temporal bounds cannot be inferred or are invalid.
    """
    bounds = configured
    if bounds is None:
        coordinate_names = tuple(str(value) for value in pool.metadata.get("coordinate_names", ()))
        metadata_bounds = pool.metadata.get("sampling_bounds")
        if metadata_bounds is not None and len(coordinate_names) == pool.inputs.shape[1]:
            named_bounds = {str(axis): (lower, upper) for axis, lower, upper in metadata_bounds}
            bounds = named_bounds.get(coordinate_names[time_axis])
    if not isinstance(bounds, Sequence) or isinstance(bounds, (str, bytes)):
        raise TypeError("Space-time `time_bounds` must be a two-value sequence.")
    values = tuple(float(value) for value in bounds)
    if len(values) != 2:
        raise ValueError("Space-time `time_bounds` must contain exactly two values.")
    return cast(tuple[float, float], values)


def _constant_templates(
    pool: HostPool,
    field_names: Sequence[str],
) -> dict[str, np.ndarray]:
    """Extract selected non-input fields that remain constant for generated coordinates.

    Args:
        pool: Host pool acting as a generated-field template.
        field_names: Explicit non-input fields included in generated batches.

    Returns:
        Host scalar or feature templates without a sample dimension.

    Raises:
        KeyError: If a requested field is absent.
        ValueError: If `inputs` is requested or a selected field varies across pool rows.
    """
    fields = pool.fields()
    templates: dict[str, np.ndarray] = {}
    for name in field_names:
        if name == "inputs":
            raise ValueError("Uniform-domain `constant_fields` cannot contain `inputs`.")
        values = fields[name]
        if values.shape[0] > 1 and not np.array_equal(values, np.broadcast_to(values[0], values.shape)):
            raise ValueError(f"Uniform-domain field `{name}` varies by row and cannot be paired with generated inputs.")
        templates[name] = np.asarray(values[0])
    return templates


__all__ = [
    "BatchSampler",
    "RandomRowSampler",
    "SamplerName",
    "SpaceTimeSampler",
    "UniformDomainSampler",
    "create_sampler",
    "sample_pool",
]
