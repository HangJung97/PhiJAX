from collections.abc import Sequence
from typing import Any

import jax
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec

from phijax.types import JaxDevice


class Strategy:
    """Define device placement operations used by the host-side trainer.

    Attributes:
        devices: Ordered devices participating in training.
        process_index: Global JAX process index.
        is_global_zero: Whether this process owns rank-zero logging side effects.
    """

    def __init__(self, devices: Sequence[JaxDevice]) -> None:
        """Initialize a strategy over explicit JAX devices.

        Args:
            devices: Non-empty ordered device collection.

        Raises:
            ValueError: If `devices` is empty.
        """
        if not devices:
            raise ValueError("A training strategy requires at least one device.")
        self.devices = tuple(devices)
        self.process_index = jax.process_index()
        self.is_global_zero = self.process_index == 0

    def place_state(self, state: Any) -> Any:
        """Place a functional training-state PyTree.

        Args:
            state: Training-state PyTree.

        Returns:
            Device-placed training state.
        """
        raise NotImplementedError

    def place_batch(self, batch: Any) -> Any:
        """Place a training-batch PyTree.

        Args:
            batch: Training-batch PyTree.

        Returns:
            Device-placed training batch.
        """
        raise NotImplementedError

    @property
    def root_device(self) -> JaxDevice:
        """Return the first strategy device addressable by this process.

        The root device is used to prepare persistent sampler state and explicit PRNG keys. Final batches still pass
        through :meth:`place_batch`, which may shard them over several devices.

        Returns:
            First process-local strategy device.

        Raises:
            RuntimeError: If the strategy contains no device addressable by this process.
        """
        for device in self.devices:
            if device.process_index == self.process_index:
                return device
        raise RuntimeError("The training strategy contains no device addressable by this JAX process.")


class SingleDeviceStrategy(Strategy):
    """Place state and batches on one explicit CPU, GPU, or TPU device."""

    def __init__(self, device: JaxDevice) -> None:
        """Initialize single-device placement.

        Args:
            device: JAX device used by the compiled training step.
        """
        super().__init__((device,))
        self.device = device

    def place_state(self, state: Any) -> Any:
        """Place the training state on the configured device.

        Args:
            state: Training-state PyTree.

        Returns:
            State placed on `device`.
        """
        return jax.device_put(state, self.device)

    def place_batch(self, batch: Any) -> Any:
        """Place the batch on the configured device.

        Args:
            batch: Training-batch PyTree.

        Returns:
            Batch placed on `device`.
        """
        return jax.device_put(batch, self.device)


class DataParallelStrategy(Strategy):
    """Replicate state and shard batch-leading axes over a named data mesh.

    The mesh may span several processes after :func:`initialize_distributed` has been called. Every process must run
    the same fit loop and encounter compiled collectives in the same order.

    Attributes:
        mesh: One-dimensional global device mesh named `data` by default.
    """

    def __init__(self, devices: Sequence[JaxDevice], axis_name: str = "data") -> None:
        """Initialize data-parallel placement.

        Args:
            devices: Global devices forming the data-parallel mesh.
            axis_name: Logical mesh-axis name.

        Raises:
            ValueError: If fewer than two devices are supplied.
        """
        if len(devices) < 2:
            raise ValueError("Data-parallel training requires at least two devices.")
        super().__init__(devices)
        self.axis_name = axis_name
        self.mesh = Mesh(np.asarray(self.devices), (axis_name,))
        self._replicated = NamedSharding(self.mesh, PartitionSpec())

    def place_state(self, state: Any) -> Any:
        """Replicate every training-state array over the mesh.

        Args:
            state: Training-state PyTree.

        Returns:
            Replicated global training state.
        """
        return jax.device_put(state, self._replicated)

    def place_batch(self, batch: Any) -> Any:
        """Shard each non-scalar batch leaf along its leading dimension.

        Args:
            batch: Training-batch PyTree whose non-scalar leading dimensions are divisible by the mesh size.

        Returns:
            Global batch with leading axes sharded over the data mesh.

        Raises:
            ValueError: If a non-scalar leading dimension is not divisible by the device count.
        """

        def place_leaf(value: Any) -> Any:
            """Place one batch leaf with replicated or leading-axis sharding.

            Args:
                value: Array-like batch leaf.

            Returns:
                Global sharded array.

            Raises:
                ValueError: If the leading dimension cannot be evenly sharded.
            """
            array = np.asarray(value) if not hasattr(value, "shape") else value
            if len(array.shape) == 0:
                if jax.process_count() > 1:
                    return jax.make_array_from_process_local_data(self._replicated, array)
                return jax.device_put(array, self._replicated)
            local_device_count = sum(device.process_index == self.process_index for device in self.devices)
            shard_count = local_device_count if jax.process_count() > 1 else len(self.devices)
            if shard_count == 0:
                raise ValueError("The data mesh contains no device addressable by this JAX process.")
            if array.shape[0] % shard_count != 0:
                raise ValueError(
                    f"Batch leading dimension {array.shape[0]} is not divisible by {shard_count} local data shards."
                )
            partition = PartitionSpec(self.axis_name, *([None] * (len(array.shape) - 1)))
            sharding = NamedSharding(self.mesh, partition)
            if jax.process_count() > 1:
                return jax.make_array_from_process_local_data(sharding, array)
            return jax.device_put(array, sharding)

        return jax.tree.map(place_leaf, batch)


def create_strategy(
    accelerator: str = "auto",
    devices: int | Sequence[int] | str = 1,
) -> Strategy:
    """Create a single-device or data-parallel strategy from visible JAX devices.

    Args:
        accelerator: One of `auto`, `cpu`, `gpu`, or `tpu`.
        devices: Device count, explicit backend-device indices, or `auto` for every matching device.

    Returns:
        Strategy over the selected devices.

    Raises:
        ValueError: If the accelerator, count, or device indices are invalid.
        RuntimeError: If the requested accelerator has no visible devices.
    """
    if accelerator not in {"auto", "cpu", "gpu", "tpu"}:
        raise ValueError("`accelerator` must be one of `auto`, `cpu`, `gpu`, or `tpu`.")
    try:
        available = tuple(jax.devices() if accelerator == "auto" else jax.devices(accelerator))
    except RuntimeError as error:
        raise RuntimeError(f"No usable `{accelerator}` JAX backend is available.") from error
    if not available:
        raise RuntimeError(f"No `{accelerator}` devices are visible to JAX.")
    if devices == "auto":
        selected = available
    elif isinstance(devices, int):
        if devices < 1 or devices > len(available):
            raise ValueError(f"Requested {devices} devices, but {len(available)} are available.")
        selected = available[:devices]
    elif isinstance(devices, Sequence) and not isinstance(devices, str):
        indices = tuple(devices)
        if not indices or any(index < 0 or index >= len(available) for index in indices):
            raise ValueError(f"Device indices must select from `0..{len(available) - 1}`.")
        selected = tuple(available[index] for index in indices)
    else:
        raise ValueError("`devices` must be a positive count, an index sequence, or `auto`.")
    return SingleDeviceStrategy(selected[0]) if len(selected) == 1 else DataParallelStrategy(selected)


def initialize_distributed(
    coordinator_address: str | None = None,
    num_processes: int | None = None,
    process_id: int | None = None,
    local_device_ids: int | Sequence[int] | None = None,
) -> None:
    """Initialize JAX multi-process execution before any backend operation.

    Args:
        coordinator_address: Coordinator `host:port`, or `None` for supported environment auto-detection.
        num_processes: Total number of participating JAX processes.
        process_id: Unique zero-based identifier for this process.
        local_device_ids: Process-local device indices made visible to this process.

    Raises:
        RuntimeError: If JAX distributed execution has already been initialized.
    """
    if jax.distributed.is_initialized():
        raise RuntimeError("JAX distributed execution is already initialized.")
    jax.distributed.initialize(
        coordinator_address=coordinator_address,
        num_processes=num_processes,
        process_id=process_id,
        local_device_ids=local_device_ids,
    )


__all__ = [
    "DataParallelStrategy",
    "SingleDeviceStrategy",
    "Strategy",
    "create_strategy",
    "initialize_distributed",
]
