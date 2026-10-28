import dataclasses
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import pytest

from phijax import __version__ as phijax_version
from phijax.balancers import BalancerState
from phijax.training import InputStatistics, OrbaxCheckpointIO, TrainState, load_input_statistics
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
        manifest = checkpoint_io._active_manager().metadata(4).custom_metadata["phijax"]
        resumed = checkpoint_io.restore(target)
        weights_only = checkpoint_io.restore_weights(target)
        restored_callback_states = checkpoint_io.restore_callback_states()

    assert float(resumed.model_state["weight"]) == 2.0
    assert float(cast(dict[str, jax.Array], resumed.optimizer_state)["slot"]) == 3.0
    assert int(resumed.step) == 4
    assert float(weights_only.model_state["weight"]) == 2.0
    assert float(cast(dict[str, jax.Array], weights_only.optimizer_state)["slot"]) == 8.0
    assert int(weights_only.step) == 0
    assert manifest["schema_version"] == 3
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
    changed = dataclasses.replace(first, model_state={"weight": jnp.ones((2,))})

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
                "schema_version": 4,
                "phijax_version": phijax_version,
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
        (
            {
                "schema_version": 3,
                "phijax_version": phijax_version,
                "step": 4,
            },
            "input-statistics layout",
        ),
        (
            {
                "schema_version": 3,
                "phijax_version": phijax_version,
                "step": 4,
                "input_statistics": {"shape": [2, 1], "dtype": "float32"},
            },
            "input-statistics layout",
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
    cast(Any, checkpoint_io)._manager = SimpleNamespace(
        metadata=lambda step: SimpleNamespace(custom_metadata=custom_metadata),
    )

    with pytest.raises(ValueError, match=message):
        checkpoint_io._validate_metadata(target, 4)


def _statistics(scale: float) -> InputStatistics:
    """Build distinguishable two-coordinate input statistics.

    Args:
        scale: Multiplier applied to the base statistics.

    Returns:
        `float32` mean and standard deviation arrays.
    """
    return InputStatistics(mean=scale * jnp.asarray([1.0, 2.0]), std=scale * jnp.asarray([0.5, 0.25]))


@pytest.mark.parametrize("template_statistics", [None, 7.0], ids=["template-without", "template-with-other"])
def test_full_restore_uses_checkpoint_input_statistics(tmp_path: Path, template_statistics: float | None) -> None:
    """Verify full restoration takes stored statistics regardless of the template, and weights-only keeps the target.

    Args:
        tmp_path: Temporary checkpoint root.
        template_statistics: Scale of the template statistics, or `None` for a template without statistics.
    """
    source = dataclasses.replace(_state(weight=2.0, slot=3.0, step=4), input_statistics=_statistics(1.0))
    target_statistics = None if template_statistics is None else _statistics(template_statistics)
    target = dataclasses.replace(_state(weight=9.0, slot=8.0, step=0), input_statistics=target_statistics)
    with OrbaxCheckpointIO(tmp_path / "checkpoints", enable_async_checkpointing=False) as checkpoint_io:
        checkpoint_io.save(source, 4)
        manifest = checkpoint_io._active_manager().metadata(4).custom_metadata["phijax"]
        resumed = checkpoint_io.restore(target)
        weights_only = checkpoint_io.restore_weights(target)

    assert manifest["schema_version"] == 3
    assert manifest["input_statistics"] == {"shape": [2], "dtype": "float32"}
    assert resumed.input_statistics is not None
    np.testing.assert_array_equal(resumed.input_statistics.mean, [1.0, 2.0])
    np.testing.assert_array_equal(resumed.input_statistics.std, [0.5, 0.25])
    assert float(weights_only.model_state["weight"]) == 2.0
    assert weights_only.input_statistics is target_statistics


def test_restore_without_stored_statistics_ignores_template_statistics(tmp_path: Path) -> None:
    """Verify a checkpoint saved without statistics restores `None` into a template that has statistics.

    Args:
        tmp_path: Temporary checkpoint root.
    """
    target = dataclasses.replace(_state(weight=9.0, slot=8.0, step=0), input_statistics=_statistics(1.0))
    with OrbaxCheckpointIO(tmp_path / "checkpoints", enable_async_checkpointing=False) as checkpoint_io:
        checkpoint_io.save(_state(weight=2.0, slot=3.0, step=4), 4)
        manifest = checkpoint_io._active_manager().metadata(4).custom_metadata["phijax"]
        resumed = checkpoint_io.restore(target)

    assert manifest["input_statistics"] is None
    assert resumed.input_statistics is None
    assert float(resumed.model_state["weight"]) == 2.0
    assert (
        checkpointing_module._state_identifier(dataclasses.replace(target, input_statistics=None))
        == manifest["state_identifier"]
    )


def test_load_input_statistics_reads_only_the_stored_statistics(tmp_path: Path) -> None:
    """Verify the transfer-learning helper returns stored statistics as host arrays, or `None`.

    Args:
        tmp_path: Temporary checkpoint roots.
    """
    with_statistics = dataclasses.replace(_state(weight=2.0, slot=3.0, step=4), input_statistics=_statistics(3.0))
    with OrbaxCheckpointIO(tmp_path / "with", enable_async_checkpointing=False) as checkpoint_io:
        checkpoint_io.save(with_statistics, 4)
    with OrbaxCheckpointIO(tmp_path / "without", enable_async_checkpointing=False) as checkpoint_io:
        checkpoint_io.save(_state(weight=2.0, slot=3.0, step=4), 4)

    statistics = load_input_statistics(tmp_path / "with")

    assert isinstance(statistics, InputStatistics)
    assert isinstance(statistics.mean, np.ndarray)
    assert statistics.mean.dtype == np.float32
    np.testing.assert_array_equal(statistics.mean, [3.0, 6.0])
    np.testing.assert_array_equal(statistics.std, [1.5, 0.75])
    assert load_input_statistics(tmp_path / "without", step=4) is None
    with pytest.raises(FileNotFoundError):
        load_input_statistics(tmp_path / "missing")


def _write_schema_2_checkpoint(directory: Path, state: TrainState, step: int) -> None:
    """Write a checkpoint in the PhiJAX 0.2.0b5 layout, which has no `input_statistics` node.

    Args:
        directory: Checkpoint root.
        state: State whose fields other than `input_statistics` are saved.
        step: Checkpoint step.
    """
    legacy_state = {
        field.name: getattr(state, field.name)
        for field in dataclasses.fields(TrainState)
        if field.name != "input_statistics"
    }
    manifest = {
        "schema_version": 2,
        "phijax_version": phijax_version,
        "step": step,
        "state_identifier": checkpointing_module._state_identifier(dataclasses.replace(state, input_statistics=None)),
        "callbacks": {},
    }
    options = ocp.CheckpointManagerOptions(enable_async_checkpointing=False)
    with ocp.CheckpointManager(directory, options=options) as manager:
        manager.save(step, args=ocp.args.StandardSave(legacy_state), custom_metadata={"phijax": manifest})


def test_schema_2_checkpoint_restores_with_template_statistics(tmp_path: Path) -> None:
    """Verify checkpoints written before statistics were stored restore and keep the template statistics.

    Args:
        tmp_path: Temporary checkpoint root.
    """
    directory = tmp_path / "checkpoints"
    _write_schema_2_checkpoint(directory, _state(weight=2.0, slot=3.0, step=4), 4)
    template = dataclasses.replace(_state(weight=9.0, slot=8.0, step=0), input_statistics=_statistics(5.0))

    with OrbaxCheckpointIO(directory, enable_async_checkpointing=False) as checkpoint_io:
        with pytest.warns(UserWarning, match="before PhiJAX stored input statistics"):
            resumed = checkpoint_io.restore(template)
        stored = checkpoint_io.restore_input_statistics()

    assert isinstance(resumed, TrainState)
    assert float(resumed.model_state["weight"]) == 2.0
    assert float(cast(dict[str, jax.Array], resumed.optimizer_state)["slot"]) == 3.0
    assert int(resumed.step) == 4
    assert resumed.input_statistics is template.input_statistics
    assert stored is None
