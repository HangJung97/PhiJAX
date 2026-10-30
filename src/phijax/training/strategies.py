import hashlib
from collections.abc import Sequence
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.sharding import AxisType, NamedSharding, PartitionSpec

from phijax.types import JaxDevice

type Accelerator = Literal["auto", "cpu", "gpu", "tpu"]
type DeviceSelection = int | Sequence[int] | Literal["auto"]


class Strategy:
    """Define device placement operations used by the host-side trainer.

    Attributes:
        process_index: Global JAX process index.
        process_count: Number of JAX processes in the run.
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
        self.process_count = jax.process_count()
        self.is_global_zero = self.process_index == 0

    def barrier(self, name: str) -> None:
        """Wait until every JAX process reaches this point.

        A single-process run returns immediately.

        Args:
            name: Identifier shared by every process for this synchronization point.
        """
        if jax.process_count() > 1:
            multihost_utils.sync_global_devices(name)

    def reduce_boolean_decisions(self, *decisions: bool, all: bool = True) -> tuple[bool, ...]:  # noqa: A002
        """Combine boolean decisions across processes, like Lightning's `Strategy.reduce_boolean_decision`.

        A single-process run returns the decisions unchanged. Otherwise one collective gathers every process's
        decisions, so every process must call this method at the same point.

        Args:
            *decisions: This process's decisions.
            all: Keep a decision only when every process made it. `False` keeps it when any process made it.

        Returns:
            Combined decisions in the input order, identical on every process.
        """
        if jax.process_count() == 1:
            return tuple(bool(decision) for decision in decisions)
        # Shape `[processes, decisions]`.
        gathered = np.asarray(multihost_utils.process_allgather(np.asarray(decisions, dtype=bool)))
        combined = gathered.all(axis=0) if all else gathered.any(axis=0)
        return tuple(bool(decision) for decision in combined)

    def process_key(self, key: jax.Array) -> jax.Array:
        """Return this process's PRNG key for host-side sampling.

        A single-process run returns `key` unchanged. In a multi-process run, `key` is a replicated global array; the
        result is this process's copy folded with :attr:`process_index`, so each process draws different samples
        while the stored key stays identical everywhere and resumes reproducibly.

        Args:
            key: Typed PRNG key from the training state.

        Returns:
            Process-local key.
        """
        if jax.process_count() == 1:
            return key
        local_key = key if key.is_fully_addressable else key.addressable_data(0)
        return jax.random.fold_in(local_key, self.process_index)

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
        # `make_mesh` may order devices for faster communication. Auto axes let `jit` propagate shardings, as the
        # compiled step does not annotate output shardings.
        self.mesh = jax.make_mesh((len(self.devices),), (axis_name,), (AxisType.Auto,), devices=self.devices)
        self._replicated = NamedSharding(self.mesh, PartitionSpec())

    def place_state(self, state: Any) -> Any:
        """Replicate every training-state array over the mesh.

        In a multi-process run, every process must hold an identical state, for example from the same `seed` or the
        same checkpoint. The state is checked once before placement.

        Args:
            state: Training-state PyTree.

        Returns:
            Replicated global training state.

        Raises:
            ValueError: If the state differs between processes.
        """
        if jax.process_count() == 1:
            return jax.device_put(state, self._replicated)
        _check_identical_across_processes(state)

        def replicate(value: Any) -> Any:
            """Build one replicated global array from this process's identical full copy.

            Args:
                value: State leaf, which every process holds in full.

            Returns:
                Global array replicated over the mesh.
            """
            if jnp.issubdtype(value.dtype, jax.dtypes.prng_key):
                data = replicate(jax.random.key_data(value))
                return jax.random.wrap_key_data(data, impl=jax.random.key_impl(value))
            return jax.make_array_from_process_local_data(self._replicated, np.asarray(value))

        # `device_put` cannot target devices owned by other processes.
        return jax.tree.map(replicate, state)

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


def _check_identical_across_processes(state: Any) -> None:
    """Verify that every process holds the same state before it is treated as replicated.

    `jax.make_array_from_process_local_data` trusts each process's copy, so differing copies would silently diverge.
    Each process gathers one SHA-256 digest per leaf instead of the full state.

    Args:
        state: Process-local training-state PyTree.

    Raises:
        ValueError: If any leaf differs between processes.
    """
    paths, leaves = zip(*jax.tree_util.tree_flatten_with_path(state)[0], strict=True)
    digests = []
    for leaf in leaves:
        values = jax.random.key_data(leaf) if jnp.issubdtype(leaf.dtype, jax.dtypes.prng_key) else leaf
        digest = hashlib.sha256(np.ascontiguousarray(np.asarray(values)).tobytes()).digest()
        digests.append(np.frombuffer(digest, dtype=np.uint8))
    # Shape `[processes, leaves, 32]`.
    gathered = np.asarray(multihost_utils.process_allgather(np.stack(digests)))
    matches = np.all(gathered == gathered[:1], axis=(0, 2))
    differing = [jax.tree_util.keystr(path) for path, match in zip(paths, matches, strict=True) if not match]
    if differing:
        raise ValueError(
            f"The training state differs between JAX processes in {differing}. Every process must start from the same "
            "state: use the same `seed`, model configuration, and checkpoint on every process."
        )


def create_strategy(
    accelerator: Accelerator = "auto",
    devices: DeviceSelection = 1,
) -> Strategy:
    """Create a single-device or data-parallel strategy from visible JAX devices.

    `devices` counts or indexes each process's local devices, as Lightning's `devices` counts devices per node. In a
    multi-process run every process selects the same local positions, and the strategy spans the selected devices of
    all processes.

    Args:
        accelerator: One of `auto`, `cpu`, `gpu`, or `tpu`.
        devices: Device count per process, explicit local device indices, or `auto` for every matching device.

    Returns:
        Strategy over the selected devices.

    Raises:
        ValueError: If the accelerator, count, or device indices are invalid, or if a process has fewer matching
            devices than the selection needs.
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
    # `jax.devices()` lists every process's devices in the same order on all processes, so grouping it by process
    # gives each process the same view of the global mesh.
    process_devices = [tuple(d for d in available if d.process_index == p) for p in range(jax.process_count())]
    local_count = len(process_devices[jax.process_index()])
    if devices == "auto":
        positions = tuple(range(local_count))
    elif isinstance(devices, int):
        if devices < 1 or devices > local_count:
            raise ValueError(f"Requested {devices} devices, but {local_count} are available.")
        positions = tuple(range(devices))
    elif isinstance(devices, Sequence) and not isinstance(devices, str):
        positions = tuple(devices)
        if not positions or any(index < 0 or index >= local_count for index in positions):
            raise ValueError(f"Device indices must select from `0..{local_count - 1}`.")
    else:
        raise ValueError("`devices` must be a positive count, an index sequence, or `auto`.")
    if any(len(group) <= max(positions) for group in process_devices):
        raise ValueError(f"Every JAX process needs at least {max(positions) + 1} `{accelerator}` devices.")
    selected = tuple(group[index] for group in process_devices for index in positions)
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
    "Accelerator",
    "DataParallelStrategy",
    "DeviceSelection",
    "SingleDeviceStrategy",
    "Strategy",
    "create_strategy",
    "initialize_distributed",
]
