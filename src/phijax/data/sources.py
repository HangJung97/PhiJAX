from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Self

import jax
import numpy as np

from phijax.data.batching import BatchSize, DevicePool, _chunk_layout, _validate_batch_size
from phijax.data.pools import HostPool
from phijax.data.samplers import BatchSampler, RandomRowSampler
from phijax.types import JaxDevice, NamedBatches


class TrainingBatchSource(Protocol):
    """Define deterministic step-indexed training and diagnostic batch delivery."""

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
    """Define finite ordered prediction batches and their reconstruction pool."""

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

    Attributes:
        samplers: Named sampling policies aligned with objective batch keys.
        batch_sizes: Per-sampler positive integer or `all` policies.
        key: Explicit root key folded with each global optimizer step.
    """

    samplers: Mapping[str, BatchSampler]
    batch_sizes: Mapping[str, BatchSize]
    key: jax.Array

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
        return self.sample(jax.random.fold_in(self.key, step))

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
        key = jax.device_put(key, self.key.sharding)
        keys = jax.random.split(key, len(self.samplers))
        batches: dict[str, DevicePool] = {}
        for sampler_key, (name, sampler) in zip(keys, self.samplers.items(), strict=True):
            policy = policies[name]
            _validate_batch_size(policy, allow_all=isinstance(sampler, RandomRowSampler))
            if policy == "all":
                batches[name] = sampler.sample_all()
            else:
                if not isinstance(policy, int):
                    raise TypeError("Validated sampler batch sizes must be integers or `all`.")
                batches[name] = sampler.sample(sampler_key, policy)
        return batches


@dataclass(frozen=True, slots=True)
class ChunkedPredictionSource:
    """Lazily construct fixed-size prediction batches from an immutable host pool.

    The source retains the complete dataset on the host. The Trainer places only the currently yielded padded batch on
    its selected device, reducing peak device memory while preserving deterministic row order.

    Attributes:
        pool: Immutable host prediction pool carrying dense reconstruction metadata.
        batch_size: Positive fixed number of rows yielded per batch.
    """

    pool: HostPool
    batch_size: int

    def __post_init__(self) -> None:
        """Validate prediction row and fixed batch sizes.

        Raises:
            ValueError: If pool rows or `batch_size` are invalid.
        """
        _chunk_layout(int(self.pool.inputs.shape[0]), self.batch_size)

    def __len__(self) -> int:
        """Return the finite number of padded prediction batches.

        Returns:
            At least one batch, including for an empty prediction pool.
        """
        batch_count, _ = _chunk_layout(int(self.pool.inputs.shape[0]), self.batch_size)
        return batch_count

    def __iter__(self) -> Iterator[dict[str, np.ndarray]]:
        """Yield host-sliced and padded prediction batches.

        Yields:
            Field mappings with a fixed leading dimension and Boolean `mask` identifying valid rows.
        """
        row_count = int(self.pool.inputs.shape[0])
        fields = self.pool.fields()
        for batch_index in range(len(self)):
            start = batch_index * self.batch_size
            stop = min(start + self.batch_size, row_count)
            valid_count = stop - start
            padding = self.batch_size - valid_count
            batch: dict[str, np.ndarray] = {}
            for name, values in fields.items():
                selected = np.asarray(values[start:stop])
                pad_width = ((0, padding), *(((0, 0),) * (selected.ndim - 1)))
                batch[name] = np.pad(selected, pad_width)
            mask = np.arange(self.batch_size) < valid_count
            batch["mask"] = mask
            yield batch


__all__ = ["ChunkedPredictionSource", "NamedBatchSource", "PredictionBatchSource", "TrainingBatchSource"]
