from types import SimpleNamespace
from typing import Any, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from phijax.training import (
    DataParallelStrategy,
    SingleDeviceStrategy,
    Strategy,
    create_strategy,
    initialize_distributed,
)
from phijax.training import strategies as strategies_module


def test_create_strategy_places_state_and_batch_on_cpu() -> None:
    """Verify explicit CPU selection produces usable single-device arrays."""
    strategy = create_strategy("cpu", 1)
    assert isinstance(strategy, SingleDeviceStrategy)
    assert strategy.root_device == jax.devices("cpu")[0]
    state = strategy.place_state({"weight": jnp.ones(2)})
    batch = strategy.place_batch({"inputs": jnp.ones((3, 2))})
    assert {device.platform for device in state["weight"].devices()} == {"cpu"}
    assert {device.platform for device in batch["inputs"].devices()} == {"cpu"}


def test_strategy_rejects_invalid_device_requests() -> None:
    """Verify device errors are raised during setup instead of a compiled update."""
    with pytest.raises(ValueError, match="accelerator"):
        create_strategy(cast(Any, "quantum"), 1)
    with pytest.raises(ValueError, match="Requested"):
        create_strategy("cpu", len(jax.devices("cpu")) + 1)
    with pytest.raises(ValueError, match="at least two"):
        DataParallelStrategy((jax.devices("cpu")[0],))


def test_base_strategy_validates_devices_and_abstract_placement(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the base strategy rejects empty placement and exposes deliberate abstract operations."""
    device = jax.devices("cpu")[0]
    with pytest.raises(ValueError, match="at least one"):
        Strategy(())
    strategy = Strategy((device,))
    with pytest.raises(NotImplementedError):
        strategy.place_state({})
    with pytest.raises(NotImplementedError):
        strategy.place_batch({})
    monkeypatch.setattr(strategy, "process_index", device.process_index + 1)
    with pytest.raises(RuntimeError, match="no device"):
        _ = strategy.root_device


def test_create_strategy_supports_auto_and_explicit_indices() -> None:
    """Verify all public device-selection forms select visible CPU devices deterministically."""
    assert isinstance(create_strategy("cpu", "auto"), SingleDeviceStrategy)
    assert isinstance(create_strategy("cpu", [0]), SingleDeviceStrategy)
    with pytest.raises(ValueError, match="positive count"):
        create_strategy("cpu", cast(Any, "one"))
    with pytest.raises(ValueError, match="Device indices"):
        create_strategy("cpu", [])
    with pytest.raises(ValueError, match="Device indices"):
        create_strategy("cpu", [-1])
    with pytest.raises(ValueError, match="Requested"):
        create_strategy("cpu", 0)


def test_create_strategy_reports_backend_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify unavailable and empty accelerator backends produce actionable runtime errors."""
    monkeypatch.setattr(jax, "devices", lambda *_: ())
    with pytest.raises(RuntimeError, match="No `gpu` devices"):
        create_strategy("gpu")

    def fail_devices(*_: object) -> tuple[()]:
        """Represent a backend that fails while JAX initializes it."""
        raise RuntimeError("missing backend")

    monkeypatch.setattr(jax, "devices", fail_devices)
    with pytest.raises(RuntimeError, match="No usable `tpu`"):
        create_strategy("tpu")


def test_create_strategy_discovers_mocked_tpu_device(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify experimental TPU selection uses the same explicit placement contract as other backends."""
    device = SimpleNamespace(platform="tpu", process_index=0, id=0)
    monkeypatch.setattr(jax, "devices", lambda backend=None: (device,) if backend in {None, "tpu"} else ())
    strategy = create_strategy("tpu", 1)

    assert isinstance(strategy, SingleDeviceStrategy)
    assert strategy.root_device is device


def test_initialize_distributed_delegates_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify distributed initialization forwards topology arguments and rejects repeated setup."""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: False)
    monkeypatch.setattr(jax.distributed, "initialize", lambda **kwargs: calls.append(kwargs))
    initialize_distributed("host:1234", 2, 1, [0])
    assert calls == [
        {
            "coordinator_address": "host:1234",
            "num_processes": 2,
            "process_id": 1,
            "local_device_ids": [0],
        }
    ]
    monkeypatch.setattr(jax.distributed, "is_initialized", lambda: True)
    with pytest.raises(RuntimeError, match="already initialized"):
        initialize_distributed()


def test_strategy_barrier_synchronizes_only_multi_process_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `barrier()` is free in a single-process run and synchronizes every process otherwise.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    synchronized: list[str] = []
    monkeypatch.setattr(strategies_module.multihost_utils, "sync_global_devices", synchronized.append)
    strategy = SingleDeviceStrategy(jax.devices("cpu")[0])

    strategy.barrier("single")
    monkeypatch.setattr(strategies_module.jax, "process_count", lambda: 2)
    strategy.barrier("multi")

    assert synchronized == ["multi"]


def test_process_key_keeps_single_process_keys_and_separates_processes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify one process samples with the stored key, while several processes fold in their process index.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    key = jax.random.key(0)
    strategy = SingleDeviceStrategy(jax.devices("cpu")[0])
    assert strategy.process_key(key) is key

    monkeypatch.setattr(strategies_module.jax, "process_count", lambda: 2)
    strategy.process_index = 1
    folded = strategy.process_key(key)

    assert jnp.array_equal(jax.random.key_data(folded), jax.random.key_data(jax.random.fold_in(key, 1)))


def test_create_strategy_selects_the_same_local_devices_on_every_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify `devices` counts and indexes each process's devices and the mesh spans all processes.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    fake_devices = [
        SimpleNamespace(process_index=process, id=2 * process + local) for process in (0, 1) for local in (0, 1)
    ]
    monkeypatch.setattr(strategies_module.jax, "devices", lambda backend=None: fake_devices)
    monkeypatch.setattr(strategies_module.jax, "process_count", lambda: 2)
    monkeypatch.setattr(strategies_module.jax, "process_index", lambda: 1)
    monkeypatch.setattr(strategies_module, "DataParallelStrategy", lambda devices: tuple(d.id for d in devices))

    assert create_strategy("cpu", devices=1) == (0, 2)
    assert create_strategy("cpu", devices=[1]) == (1, 3)
    assert create_strategy("cpu", devices="auto") == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="Requested 3 devices, but 2 are available"):
        create_strategy("cpu", devices=3)

    # Process 0 has two devices, but process 1 has only one.
    uneven = fake_devices[:3]
    monkeypatch.setattr(strategies_module.jax, "devices", lambda backend=None: uneven)
    monkeypatch.setattr(strategies_module.jax, "process_index", lambda: 0)
    with pytest.raises(ValueError, match="Every JAX process needs at least 2"):
        create_strategy("cpu", devices=[1])


def test_data_parallel_mesh_uses_make_mesh_with_auto_axes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the data mesh comes from `jax.make_mesh` with explicit Auto axes, as the compiled step relies on them.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    calls: list[tuple[Any, ...]] = []
    original_make_mesh = jax.make_mesh

    def make_mesh(axis_sizes: Any, axis_names: Any, axis_types: Any, *, devices: Any) -> Any:
        """Record mesh arguments and build a real single-device mesh.

        Args:
            axis_sizes: Requested axis sizes.
            axis_names: Requested axis names.
            axis_types: Requested axis types.
            devices: Requested mesh devices.

        Returns:
            One-device CPU mesh.
        """
        calls.append((tuple(axis_sizes), tuple(axis_names), tuple(axis_types), len(devices)))
        return original_make_mesh((1,), axis_names, axis_types, devices=jax.devices("cpu")[:1])

    monkeypatch.setattr(strategies_module.jax, "make_mesh", make_mesh)
    cpu = jax.devices("cpu")[0]
    DataParallelStrategy((cpu, cpu), axis_name="batch")

    assert calls == [((2,), ("batch",), (strategies_module.AxisType.Auto,), 2)]


def test_multi_process_state_check_names_differing_leaves(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify identical process states pass and a differing leaf, here a PRNG key, is reported by path.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    state = {"key": jax.random.key(0), "weight": jnp.ones(2)}
    differing_leaf: list[int] = []

    def allgather(digests: np.ndarray) -> np.ndarray:
        """Stack this process's digests with a simulated second process whose selected leaves differ.

        Args:
            digests: This process's digests with shape `[leaves, 32]`.

        Returns:
            Digests of both processes with shape `[2, leaves, 32]`.
        """
        other = digests.copy()
        other[differing_leaf] ^= 1
        return np.stack([digests, other])

    monkeypatch.setattr(strategies_module.multihost_utils, "process_allgather", allgather)
    strategies_module._check_identical_across_processes(state)

    differing_leaf.append(0)
    with pytest.raises(ValueError, match=r"differs between JAX processes in \[\"\['key'\]\"\]"):
        strategies_module._check_identical_across_processes(state)


def test_reduce_boolean_decisions_combines_processes_with_any_or_all(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify decisions pass through in one process and combine across processes in a single gather.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    strategy = SingleDeviceStrategy(jax.devices("cpu")[0])
    assert strategy.reduce_boolean_decisions(True, False) == (True, False)

    gathers: list[np.ndarray] = []

    def allgather(decisions: np.ndarray) -> np.ndarray:
        """Stack this process's decisions with a simulated second process that decided the opposite.

        Args:
            decisions: This process's decisions.

        Returns:
            Decisions of both processes with shape `[2, decisions]`.
        """
        gathers.append(decisions)
        return np.stack([decisions, ~decisions])

    monkeypatch.setattr(strategies_module.jax, "process_count", lambda: 2)
    monkeypatch.setattr(strategies_module.multihost_utils, "process_allgather", allgather)

    assert strategy.reduce_boolean_decisions(True, False, all=False) == (True, True)
    assert strategy.reduce_boolean_decisions(True, False) == (False, False)
    assert len(gathers) == 2
