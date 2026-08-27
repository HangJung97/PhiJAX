from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import pytest

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
        step=jnp.asarray(step, jnp.int32),
        loss_scale=jnp.asarray(1.0, jnp.float32),
        finite_steps=jnp.asarray(step, jnp.int32),
    )


def test_orbax_checkpoint_supports_resume_and_weights_only_loading(tmp_path: Path) -> None:
    """Verify exact resume restores all fields while weight loading preserves fresh run state."""
    source = _state(weight=2.0, slot=3.0, step=4)
    target = _state(weight=9.0, slot=8.0, step=0)
    with OrbaxCheckpointIO(tmp_path / "checkpoints", enable_async_checkpointing=False) as checkpoint_io:
        assert checkpoint_io.save(source, 4, {"train/loss": 1.0}) is True
        checkpoint_io.wait_until_finished()
        manifest = checkpoint_io._manager.metadata(4).custom_metadata["phijax"]
        resumed = checkpoint_io.restore(target)
        weights_only = checkpoint_io.restore_weights(target)

    assert float(resumed.model_state["weight"]) == 2.0
    assert float(resumed.optimizer_state["slot"]) == 3.0
    assert int(resumed.step) == 4
    assert float(weights_only.model_state["weight"]) == 2.0
    assert float(weights_only.optimizer_state["slot"]) == 8.0
    assert int(weights_only.step) == 0
    assert manifest["schema_version"] == 1
    assert manifest["step"] == 4
    assert manifest["phijax_version"] == "0.1.0b1"


def test_checkpoint_state_identifier_tracks_structure_not_values() -> None:
    """Verify checkpoint identifiers are stable across values and change with leaf shapes."""
    first = _state(weight=1.0, slot=2.0, step=0)
    second = _state(weight=3.0, slot=4.0, step=5)
    changed = first.replace(model_state={"weight": jnp.ones((2,))})

    assert checkpointing_module._state_identifier(first) == checkpointing_module._state_identifier(second)
    assert checkpointing_module._state_identifier(first) != checkpointing_module._state_identifier(changed)


@pytest.mark.parametrize("producer", ["0.2.0", None])
def test_checkpoint_manifest_rejects_incompatible_producers(producer: str | None) -> None:
    """Verify restore validation rejects absent and cross-minor manifests.

    Args:
        producer: Incompatible producer version or missing-manifest marker.
    """
    target = _state(weight=1.0, slot=2.0, step=0)
    checkpoint_io = OrbaxCheckpointIO.__new__(OrbaxCheckpointIO)
    custom_metadata = (
        {}
        if producer is None
        else {
            "phijax": {
                "schema_version": 1,
                "phijax_version": producer,
                "step": 4,
                "state_identifier": checkpointing_module._state_identifier(target),
            }
        }
    )
    checkpoint_io._manager = SimpleNamespace(
        metadata=lambda step: SimpleNamespace(custom_metadata=custom_metadata),
    )

    with pytest.raises(ValueError, match=r"manifest|cannot be restored"):
        checkpoint_io._validate_metadata(target, 4)
