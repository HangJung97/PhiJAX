import jax
import jax.numpy as jnp
import pytest

from phijax.training import (
    DataParallelStrategy,
    SingleDeviceStrategy,
    Strategy,
    create_strategy,
    initialize_distributed,
)


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
        create_strategy("quantum", 1)
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
        create_strategy("cpu", "one")
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
