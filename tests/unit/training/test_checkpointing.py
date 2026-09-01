from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest

from phijax import __version__ as phijax_version
from phijax.balancers import BalancerState
from phijax.training import OrbaxCheckpointIO, TrainState
from phijax.training import checkpointing as checkpointing_module


def _state(weight: float, slot: float, step: int) -> TrainState:
    """Build a distinguishable full state for save and restore checks.

    Args:
        weight: Scalar model parameter.
        slot: Scalar optimizer accumulator.
        step: Optimizer step.

    Returns:
        Complete functional training state.
    """
    return TrainState(
        model_state={"weight": jnp.asarray(weight)},
        optimizer_state={"slot": jnp.asarray(slot)},
        balancer_state=BalancerState(weights=jnp.ones(1), traces=jnp.zeros(1)),
        rng_key=jax.random.key(step),
        sampling_key=jax.random.key(step + 1),
        balancer_key=jax.random.key(step + 2),
        step=jnp.asarray(step, jnp.int32),
        loss_scale=jnp.asarray(1.0, jnp.float32),
        finite_steps=jnp.asarray(step, jnp.int32),
    )


def test_orbax_checkpoint_supports_resume_and_weights_only_loading(tmp_path: Path) -> None:
    """Verify exact resume restores all fields while weight loading preserves fresh run state."""
    source = _state(weight=2.0, slot=3.0, step=4)
    target = _state(weight=9.0, slot=8.0, step=0)
    callback_states = {"phijax.callbacks.Example:0": {"last_step": 3}}
    with OrbaxCheckpointIO(tmp_path / "checkpoints", enable_async_checkpointing=False) as checkpoint_io:
        assert checkpoint_io.save(source, 4, {"train/loss": 1.0}, callback_states=callback_states) is True
        checkpoint_io.wait_until_finished()
        manifest = checkpoint_io._manager.metadata(4).custom_metadata["phijax"]
        resumed = checkpoint_io.restore(target)
        weights_only = checkpoint_io.restore_weights(target)
        restored_callback_states = checkpoint_io.restore_callback_states()

    assert float(resumed.model_state["weight"]) == 2.0
    assert float(resumed.optimizer_state["slot"]) == 3.0
    assert int(resumed.step) == 4
    assert float(weights_only.model_state["weight"]) == 2.0
    assert float(weights_only.optimizer_state["slot"]) == 8.0
    assert int(weights_only.step) == 0
    assert manifest["schema_version"] == 2
    assert manifest["step"] == 4
    assert manifest["phijax_version"] == phijax_version
    assert manifest["callbacks"] == callback_states
    assert restored_callback_states == callback_states


def test_orbax_checkpoint_manager_opens_lazily_and_reopens(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify stage cleanup can recreate the Orbax manager without explicit Trainer cleanup.

    Args:
        monkeypatch: Pytest attribute patch helper.
        tmp_path: Temporary checkpoint-root parent.
    """
    managers: list[MagicMock] = []

    def create_manager(*args: object, **kwargs: object) -> MagicMock:
        """Create and record one mock Orbax manager.

        Args:
            *args: Positional manager arguments.
            **kwargs: Keyword manager arguments.

        Returns:
            Fresh mock manager.
        """
        del args, kwargs
        manager = MagicMock()
        managers.append(manager)
        return manager

    monkeypatch.setattr(checkpointing_module.ocp, "CheckpointManager", create_manager)
    checkpoint_io = OrbaxCheckpointIO(tmp_path / "checkpoints")

    assert checkpoint_io._manager is None
    checkpoint_io.open()
    checkpoint_io.open()
    assert len(managers) == 1

    checkpoint_io.close()
    checkpoint_io.close()
    managers[0].close.assert_called_once_with()
    assert checkpoint_io._manager is None

    checkpoint_io.open()
    assert len(managers) == 2
    checkpoint_io.close()
    managers[1].close.assert_called_once_with()


def test_checkpoint_state_identifier_tracks_structure_not_values() -> None:
    """Verify checkpoint identifiers are stable across values and change with leaf shapes."""
    first = _state(weight=1.0, slot=2.0, step=0)
    second = _state(weight=3.0, slot=4.0, step=5)
    changed = first.replace(model_state={"weight": jnp.ones((2,))})

    assert checkpointing_module._state_identifier(first) == checkpointing_module._state_identifier(second)
    assert checkpointing_module._state_identifier(first) != checkpointing_module._state_identifier(changed)


@pytest.mark.parametrize(
    ("metadata", "message"),
    [
        ({}, "manifest"),
        (
            {
                "schema_version": 1,
                "phijax_version": "0.2.0b1",
                "step": 4,
            },
            "schema",
        ),
        (
            {
                "schema_version": 2,
                "phijax_version": "0.1.0b1",
                "step": 4,
            },
            "cannot be restored",
        ),
    ],
)
def test_checkpoint_manifest_rejects_incompatible_producers(metadata: dict[str, object], message: str) -> None:
    """Verify restore validation rejects absent, old-schema, and cross-minor manifests.

    Args:
        metadata: Candidate PhiJAX checkpoint manifest fields.
        message: Expected incompatibility-message fragment.
    """
    target = _state(weight=1.0, slot=2.0, step=0)
    checkpoint_io = OrbaxCheckpointIO.__new__(OrbaxCheckpointIO)
    custom_metadata = {} if not metadata else {"phijax": metadata | {"state_identifier": "unused"}}
    checkpoint_io._manager = SimpleNamespace(
        metadata=lambda step: SimpleNamespace(custom_metadata=custom_metadata),
    )

    with pytest.raises(ValueError, match=message):
        checkpoint_io._validate_metadata(target, 4)
