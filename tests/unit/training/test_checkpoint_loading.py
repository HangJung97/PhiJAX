from pathlib import Path
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest

from phijax.balancers import BalancerState
from phijax.training import TrainState
from phijax.training import checkpointing as checkpointing_module


def _state(step: int) -> TrainState:
    """Build a minimal distinguishable checkpoint target.

    Args:
        step: Scalar optimizer step stored in the state.

    Returns:
        Complete synthetic training state.
    """
    return TrainState(
        model_state={"weight": jnp.asarray(float(step))},
        optimizer_state=(),
        balancer_state=BalancerState(weights=jnp.ones(1), traces=jnp.zeros(1)),
        rng_key=jax.random.key(step),
        sampling_key=jax.random.key(step + 1),
        balancer_key=jax.random.key(step + 2),
        step=jnp.asarray(step, jnp.int32),
        loss_scale=jnp.asarray(1.0, jnp.float32),
        finite_steps=jnp.asarray(0, jnp.int32),
    )


def test_restore_checkpoint_leaves_fresh_training_state_untouched(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a null checkpoint path performs no storage access and preserves object identity.

    Args:
        monkeypatch: Pytest attribute patch helper.
    """
    checkpoint_io = MagicMock()
    monkeypatch.setattr(checkpointing_module, "OrbaxCheckpointIO", checkpoint_io)
    target = _state(0)
    restored = checkpointing_module.restore_checkpoint(target, None)
    assert restored is target
    checkpoint_io.assert_not_called()


def test_restore_checkpoint_selects_full_resume_or_weights_only_transfer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Verify one optional path supports both complete resumption and transfer initialization.

    Args:
        monkeypatch: Pytest attribute patch helper.
        tmp_path: Synthetic checkpoint-root path.
    """
    target = _state(0)
    resumed = _state(4)
    transferred = _state(0).replace(model_state={"weight": jnp.asarray(4.0)})
    manager = MagicMock()
    manager.__enter__.return_value = manager
    manager.restore.return_value = resumed
    manager.restore_weights.return_value = transferred
    checkpoint_io = MagicMock(return_value=manager)
    monkeypatch.setattr(checkpointing_module, "OrbaxCheckpointIO", checkpoint_io)

    full_result = checkpointing_module.restore_checkpoint(target, tmp_path, step=4)
    weights_result = checkpointing_module.restore_checkpoint(target, tmp_path, weights_only=True)

    assert full_result is resumed
    assert weights_result is transferred
    manager.restore.assert_called_once_with(target, 4)
    manager.restore_weights.assert_called_once_with(target, None)
    checkpoint_io.assert_called_with(tmp_path, max_to_keep=None, enable_async_checkpointing=False)


@pytest.mark.parametrize(
    ("ckpt_path", "weights_only", "step"),
    [
        (None, True, None),
        (None, False, 1),
        ("", False, None),
        ("   ", False, None),
        ("checkpoints", False, -1),
        ("checkpoints", False, True),
    ],
)
def test_restore_checkpoint_rejects_incomplete_or_invalid_selection(
    ckpt_path: str | None,
    weights_only: bool,
    step: int | None,
) -> None:
    """Verify restore-specific options require a valid checkpoint source.

    Args:
        ckpt_path: Invalid or absent checkpoint path.
        weights_only: Whether the case requests transfer loading.
        step: Invalid or pathless checkpoint step.
    """
    with pytest.raises(ValueError, match="ckpt_"):
        checkpointing_module.restore_checkpoint(_state(0), ckpt_path, weights_only=weights_only, step=step)
