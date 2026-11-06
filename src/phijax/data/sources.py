from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, Self, cast

import jax
import numpy as np

from phijax.data.batching import BatchSize, DevicePool, _chunk_layout, _validate_batch_size
from phijax.data.pools import HostPool
from phijax.data.samplers import BatchSampler, RandomRowSampler
from phijax.types import JaxDevice, NamedBatches


class TrainingBatchSource(Protocol):
    """Define deterministic step-indexed training and diagnostic batch delivery.

    Batch sizes are per device. A source may also define an optional `for_devices(count: int) -> Self` method that
    returns a copy drawing `count` times every integer batch size, as :meth:`NamedBatchSource.for_devices` does. The
    Trainer calls it before :meth:`prepare` with this process's device count, because each process draws its own batch
    and splits it over its devices. Diagnostic sizes passed to :meth:`sample` are not scaled. A source without
    `for_devices` is used as it is, so its batches are the rows of the whole process and their leading dimension must
    divide by the process's device count.
    """

    def prepare(self, device: JaxDevice) -> Self:
        """Prepare persistent source state for one trainer-selected device.

        Args:
            device: Trainer Strategy's process-local root device.

        Returns:
            Device-ready source preserving sampling policy and deterministic key behavior.
        """
        ...

    def __call__(self, step: int) -> NamedBatches:
        """Produce one named training batch for a global optimizer step.

        Args:
            step: Non-negative global optimizer step, including a restored checkpoint offset.

        Returns:
            Named fixed-structure training batches.
        """
        ...

    def sample(
        self,
        key: jax.Array,
        batch_sizes: Mapping[str, BatchSize] | None = None,
    ) -> NamedBatches:
        """Produce explicitly keyed batches with optional size overrides.

        Args:
            key: Explicit root sampling key.
            batch_sizes: Optional per-stream batch-size overrides used by diagnostics such as exact NTK balancing.

        Returns:
            Named fixed-structure batches.
        """
        ...


class PredictionBatchSource(Protocol):
    """Define finite ordered prediction batches and their reconstruction pool.

    Chunk sizes are per device. A source may also define an optional `for_devices(count: int) -> Self` method that
    returns a copy with `count` times larger chunks, as :meth:`ChunkedPredictionSource.for_devices` does. The Trainer
    calls it with the total device count, because every process iterates the same chunks and splits each one over all
    devices. A source without `for_devices` is used as it is; its chunks are padded to divide over the devices.
    """

    @property
    def pool(self) -> HostPool:
        """Return the immutable host pool used to reconstruct flat predictions."""
        ...

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        """Iterate over fixed-size host-backed padded prediction batches."""
        ...

    def __len__(self) -> int:
        """Return the finite number of prediction batches."""
        ...


@dataclass(frozen=True, slots=True)
class NamedBatchSource:
    """Compose named immutable samplers into deterministic step-indexed batches.

    Every sampler with an integer batch size is drawn in one compiled call per step, so a step costs one dispatch
    instead of one per array operation. Samplers must therefore be traceable by :func:`jax.jit`. A sampler with the
    `all` policy returns its complete pool outside the compiled call, which avoids copying the pool on every step.

    Attributes:
        samplers: Named sampling policies aligned with objective batch keys.
        batch_sizes: Per-sampler positive integer or `all` policies. Integer sizes are rows per device.
        key: Explicit root key folded with each global optimizer step.
    """

    samplers: Mapping[str, BatchSampler]
    batch_sizes: Mapping[str, BatchSize]
    key: jax.Array
    # Compiled draws keyed by their integer batch sizes and whether they fold a step into the key. Each prepared or
    # rescaled copy starts empty, because its compiled functions close over that copy's samplers.
    _draws: dict[tuple[Any, ...], Callable[..., dict[str, DevicePool]]] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    @classmethod
    def from_pools(
        cls,
        pools: Mapping[str, HostPool],
        batch_sizes: Mapping[str, BatchSize],
        key: jax.Array,
        *,
        names: tuple[str, ...] | None = None,
        replace: bool = True,
        sort_axes: Mapping[str, int | None] | None = None,
    ) -> Self:
        """Construct aligned finite-row samplers from named host pools.

        Args:
            pools: Finite host pools keyed by application batch name.
            batch_sizes: Sampling policies keyed by the same names.
            key: Explicit root key folded with each optimizer step.
            names: Optional stable subset and ordering. The batch-size declaration order is used when omitted.
            replace: Whether sampled rows may repeat.
            sort_axes: Optional input-coordinate sort axis per selected pool.

        Returns:
            Deterministic named source containing one :class:`RandomRowSampler` per selected pool.

        Raises:
            KeyError: If a selected name is absent from `pools` or `batch_sizes`.
            ValueError: If selected names are empty or duplicated.
        """
        selected = tuple(batch_sizes) if names is None else tuple(names)
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("`names` must contain unique non-empty pool names.")
        if any(not name or not name.strip() for name in selected):
            raise ValueError("`names` must contain unique non-empty pool names.")
        axes = dict(sort_axes or {})
        samplers = {
            name: RandomRowSampler(pools[name].fields(), replace=replace, sort_axis=axes.get(name)) for name in selected
        }
        policies = {name: batch_sizes[name] for name in selected}
        return cls(samplers, policies, key)

    def __post_init__(self) -> None:
        """Validate sampler names and batch-size policies.

        Raises:
            KeyError: If a sampler has no batch-size policy.
            ValueError: If no samplers are supplied or a policy is invalid for its sampler.
        """
        if not self.samplers:
            raise ValueError("At least one named sampler is required.")
        for name, sampler in self.samplers.items():
            _validate_batch_size(self.batch_sizes[name], allow_all=isinstance(sampler, RandomRowSampler))

    def for_devices(self, count: int) -> Self:
        """Return a copy that draws every integer batch size once per device.

        Batch sizes are per device, as in Lightning. The Trainer calls this with the number of devices that share
        each sampled batch, so every device receives the configured rows. `all` policies keep the whole pool.

        Args:
            count: Positive number of devices that split each batch.

        Returns:
            This source when `count` is `1`, otherwise a copy with scaled batch sizes.
        """
        if count == 1:
            return self
        sizes = {name: size if size == "all" else size * count for name, size in self.batch_sizes.items()}
        return type(self)(self.samplers, sizes, self.key)

    def prepare(self, device: JaxDevice) -> Self:
        """Place persistent sampler arrays and the root key on a trainer-selected device.

        Preparing once retains device-side row selection and generated-coordinate sampling without requiring an
        application DataModule to know which accelerator the Trainer selected.

        Args:
            device: Trainer Strategy's process-local root device.

        Returns:
            Immutable device-ready source.
        """
        samplers = {name: sampler.prepare(device) for name, sampler in self.samplers.items()}
        key = self.key if self.key.devices() == {device} else jax.device_put(self.key, device)
        if key is self.key and all(samplers[name] is sampler for name, sampler in self.samplers.items()):
            return self
        return type(self)(samplers, self.batch_sizes, key)

    def __call__(self, step: int) -> dict[str, DevicePool]:
        """Sample every named policy for one global optimizer step.

        Args:
            step: Non-negative global optimizer step, including any restored checkpoint offset.

        Returns:
            Named fixed-structure training batches.

        Raises:
            ValueError: If `step` is negative.
        """
        if step < 0:
            raise ValueError("`step` must be non-negative.")
        return self._draw(self.batch_sizes, self.key, step)

    def sample(
        self,
        key: jax.Array,
        batch_sizes: Mapping[str, BatchSize] | None = None,
    ) -> dict[str, DevicePool]:
        """Sample named policies with an explicit key and optional size overrides.

        Args:
            key: Explicit root key split in sampler declaration order.
            batch_sizes: Optional replacement policies, used for fixed balancer diagnostic batches.

        Returns:
            Named batches aligned with sampler declaration order.

        Raises:
            KeyError: If a sampler has no selected batch-size policy.
            ValueError: If an override is invalid or requests `all` from a generated sampler.
        """
        policies = self.batch_sizes if batch_sizes is None else batch_sizes
        # Explicit diagnostic keys follow the source's prepared placement so random operations and sampler storage
        # never silently cross devices.
        return self._draw(policies, jax.device_put(key, self.key.sharding))

    def _draw(
        self,
        policies: Mapping[str, BatchSize],
        key: jax.Array,
        step: int | None = None,
    ) -> dict[str, DevicePool]:
        """Draw every named batch, sampling integer policies in one compiled call.

        Args:
            policies: Batch-size policy for every sampler.
            key: Root key, folded with `step` when one is given, then split in sampler declaration order.
            step: Optional global step folded into `key` inside the compiled call.

        Returns:
            Named batches aligned with sampler declaration order.

        Raises:
            KeyError: If a sampler has no selected batch-size policy.
            ValueError: If a policy is invalid or requests `all` from a generated sampler.
        """
        for name, sampler in self.samplers.items():
            _validate_batch_size(policies[name], allow_all=isinstance(sampler, RandomRowSampler))
        sizes = tuple((name, cast(int, policies[name])) for name in self.samplers if policies[name] != "all")
        sampled: dict[str, DevicePool] = {}
        if sizes:
            draw = self._compiled_draw(sizes, fold_step=step is not None)
            sampled = draw(key, 0 if step is None else step)
        return {
            name: sampled[name] if name in sampled else sampler.sample_all() for name, sampler in self.samplers.items()
        }

    def _compiled_draw(
        self,
        sizes: tuple[tuple[str, int], ...],
        *,
        fold_step: bool,
    ) -> Callable[[jax.Array, int], dict[str, DevicePool]]:
        """Return the cached compiled draw for one set of integer batch sizes.

        Args:
            sizes: `(name, batch_size)` pairs for the samplers drawn inside the compiled call.
            fold_step: Whether the compiled call folds its step argument into the key before splitting it.

        Returns:
            Compiled function mapping a root key and step to the sized batches.
        """
        cache_key = (sizes, fold_step)
        draw = self._draws.get(cache_key)
        if draw is not None:
            return draw
        positions = {name: index for index, name in enumerate(self.samplers)}

        def draw_batches(key: jax.Array, step: jax.Array) -> dict[str, DevicePool]:
            """Fold the optional step, split one key per sampler, and draw the sized batches.

            Args:
                key: Root key.
                step: Traced global step, ignored unless `fold_step` is set.

            Returns:
                Sized batches keyed by sampler name.
            """
            root = jax.random.fold_in(key, step) if fold_step else key
            # Every sampler receives the key at its declaration position, including samplers drawn with `all`, so
            # adding or removing an `all` pool never shifts another pool's random stream.
            keys = jax.random.split(root, len(positions))
            return {name: self.samplers[name].sample(keys[positions[name]], size) for name, size in sizes}

        # Sampler arrays are closed over: JAX passes them to the compiled call as device buffers, not as constants.
        draw = self._draws[cache_key] = jax.jit(draw_batches)
        return draw


@dataclass(frozen=True, slots=True)
class ChunkedPredictionSource:
    """Lazily construct fixed-size prediction batches from an immutable host pool.

    The source retains the complete dataset on the host. The Trainer places only the currently yielded padded batch on
    its selected device, reducing peak device memory while preserving deterministic row order.

    Attributes:
        pool: Immutable host prediction pool carrying dense reconstruction metadata.
        batch_size: Positive fixed number of rows yielded per batch, for each device that shares it, or `"all"` for one
            batch holding every pool row, split over the devices that share it.
    """

    pool: HostPool
    batch_size: BatchSize

    def __post_init__(self) -> None:
        """Validate prediction row and fixed batch sizes and the reserved `valid_predict_mask` field.

        Raises:
            ValueError: If pool rows or `batch_size` are invalid, or if the pool has an array field named
                `valid_predict_mask`.
        """
        _chunk_layout(int(self.pool.inputs.shape[0]), self._rows_per_batch)
        if "valid_predict_mask" in self.pool.fields():
            raise ValueError(
                "Prediction pools cannot have an auxiliary array named `valid_predict_mask`; ChunkedPredictionSource "
                "uses that name for its padding mask. Rename the field."
            )

    def for_devices(self, count: int) -> Self:
        """Return a copy whose chunks hold `batch_size` rows for each device.

        The chunk size is per device, as in Lightning. The Trainer calls this with the number of devices that share
        each chunk.

        Args:
            count: Positive number of devices that split each chunk.

        Returns:
            This source when `count` is `1` or `batch_size` is `"all"`, otherwise a copy with a scaled chunk size.
        """
        if count == 1 or self.batch_size == "all":
            return self
        return type(self)(self.pool, cast(int, self.batch_size) * count)

    @property
    def _rows_per_batch(self) -> int:
        """Resolve `batch_size` to the fixed leading dimension of every yielded batch.

        Returns:
            `batch_size`, or the pool row count for `"all"`. An empty pool keeps one row, which is padding.
        """
        if self.batch_size == "all":
            return max(1, int(self.pool.inputs.shape[0]))
        return cast(int, self.batch_size)

    def __len__(self) -> int:
        """Return the finite number of padded prediction batches.

        Returns:
            At least one batch, including for an empty prediction pool.
        """
        batch_count, _ = _chunk_layout(int(self.pool.inputs.shape[0]), self._rows_per_batch)
        return batch_count

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """Yield host-sliced and padded prediction batches.

        Yields:
            Field mappings with a fixed leading dimension and a Boolean `valid_predict_mask` that is `True` for pool
            rows and `False` for padding.
        """
        row_count = int(self.pool.inputs.shape[0])
        rows = self._rows_per_batch
        fields = self.pool.fields()
        for batch_index in range(len(self)):
            start = batch_index * rows
            stop = min(start + rows, row_count)
            valid_count = stop - start
            padding = rows - valid_count
            batch: dict[str, np.ndarray] = {}
            for name, values in fields.items():
                selected = np.asarray(values[start:stop])
                pad_width = ((0, padding), *(((0, 0),) * (selected.ndim - 1)))
                batch[name] = np.pad(selected, pad_width)
            batch["valid_predict_mask"] = np.arange(rows) < valid_count
            yield batch


__all__ = ["ChunkedPredictionSource", "NamedBatchSource", "PredictionBatchSource", "TrainingBatchSource"]
