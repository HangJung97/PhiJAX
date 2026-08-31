from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import jax
import orbax.checkpoint as ocp

from phijax.training.state import TrainState

_CHECKPOINT_SCHEMA_VERSION = 2


def _state_identifier(state: TrainState) -> str:
    """Create a stable identifier from state paths, shapes, and dtypes.

    Args:
        state: Training-state PyTree used as a save value or restore template.

    Returns:
        Hexadecimal SHA-256 digest describing the state structure.
    """
    entries = []
    for path, leaf in jax.tree_util.tree_flatten_with_path(state)[0]:
        shape = tuple(int(size) for size in getattr(leaf, "shape", ()))
        dtype = str(getattr(leaf, "dtype", type(leaf).__qualname__))
        entries.append((jax.tree_util.keystr(path), shape, dtype))
    payload = json.dumps(entries, separators=(",", ":"), sort_keys=True).encode()
    return sha256(payload).hexdigest()


def _version_line(value: str) -> tuple[int, int]:
    """Extract the major and minor release line from a version string.

    Args:
        value: PEP 440-compatible package version beginning with numeric major and minor fields.

    Returns:
        Numeric major and minor version.

    Raises:
        ValueError: If the version does not begin with two numeric fields.
    """
    fields = value.split(".", maxsplit=2)
    if len(fields) < 2 or not fields[0].isdigit() or not fields[1].isdigit():
        raise ValueError(f"Invalid PhiJAX checkpoint producer version `{value}`.")
    return int(fields[0]), int(fields[1])


def _checkpoint_metadata(
    state: TrainState,
    step: int,
    callback_states: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build versioned metadata stored with one checkpoint step.

    Args:
        state: Complete state submitted for saving.
        step: Checkpoint step associated with the state.
        callback_states: Optional JSON-compatible callback state mapping.

    Returns:
        JSON-compatible PhiJAX checkpoint metadata.
    """
    return {
        "phijax": {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "phijax_version": version("phijax"),
            "step": step,
            "state_identifier": _state_identifier(state),
            "callbacks": dict(callback_states or {}),
        }
    }


class OrbaxCheckpointIO:
    """Manage lazily opened asynchronous full-state checkpoints with Orbax.

    Attributes:
        directory: Absolute checkpoint-root directory.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        max_to_keep: int | None = 3,
        enable_async_checkpointing: bool = True,
    ) -> None:
        """Initialize an Orbax checkpoint manager.

        Args:
            directory: Checkpoint-root directory.
            max_to_keep: Maximum recent checkpoints retained, or `None` to retain all.
            enable_async_checkpointing: Whether saves may finish in a background thread.

        Raises:
            ValueError: If `max_to_keep` is not positive or `None`.
        """
        if max_to_keep is not None and max_to_keep < 1:
            raise ValueError("`max_to_keep` must be positive or `None`.")
        self.directory = Path(directory).expanduser().resolve()
        self._options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            enable_async_checkpointing=enable_async_checkpointing,
        )
        self._manager: ocp.CheckpointManager | None = None

    def open(self) -> None:
        """Open the Orbax manager when it is not already active."""
        if self._manager is None:
            self._manager = ocp.CheckpointManager(self.directory, options=self._options)

    def _active_manager(self) -> ocp.CheckpointManager:
        """Return an active manager, opening it lazily when required.

        Returns:
            Active Orbax checkpoint manager.

        Raises:
            RuntimeError: If Orbax does not create a checkpoint manager.
        """
        self.open()
        manager = self._manager
        if manager is None:
            raise RuntimeError("Orbax did not create a checkpoint manager.")
        return manager

    @property
    def latest_step(self) -> int | None:
        """Return the latest committed checkpoint step.

        Returns:
            Latest step, or `None` when no checkpoint exists.
        """
        return self._active_manager().latest_step()

    def save(
        self,
        state: TrainState,
        step: int,
        metrics: Mapping[str, float] | None = None,
        *,
        force: bool = False,
        callback_states: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> bool:
        """Submit a complete training state for checkpointing.

        Args:
            state: Full functional training state.
            step: Unique checkpoint step.
            metrics: Optional host scalar metrics used by Orbax retention policies.
            force: Whether to save independently of manager policies.
            callback_states: Optional callback states restored before fit-start hooks.

        Returns:
            Whether Orbax initiated a save.
        """
        return self._active_manager().save(
            step,
            args=ocp.args.StandardSave(state),
            metrics=dict(metrics or {}),
            force=force,
            custom_metadata=_checkpoint_metadata(state, step, callback_states),
        )

    @property
    def steps(self) -> tuple[int, ...]:
        """Return committed checkpoint steps in ascending order.

        Returns:
            Ordered committed steps.
        """
        return tuple(sorted(self._active_manager().all_steps()))

    def checkpoint_path(self, step: int) -> Path:
        """Return the filesystem path associated with a checkpoint step.

        Args:
            step: Committed checkpoint step.

        Returns:
            Expected Orbax step directory.
        """
        return self.directory / str(step)

    def delete(self, step: int) -> None:
        """Delete one committed checkpoint and wait for completion.

        Args:
            step: Checkpoint step to remove.
        """
        manager = self._active_manager()
        manager.delete(step)
        manager.wait_until_finished()

    def restore(self, target: TrainState, step: int | None = None) -> TrainState:
        """Restore a complete state for exact training resumption.

        Args:
            target: Abstract or concrete state defining the expected structure and sharding.
            step: Checkpoint step, or `None` for the latest committed step.

        Returns:
            Restored model, optimizer, balancer, PRNG, step, and precision state.

        Raises:
            FileNotFoundError: If no requested checkpoint exists.
        """
        resolved_step = self.latest_step if step is None else step
        if resolved_step is None:
            raise FileNotFoundError(f"No checkpoints exist below `{self.directory}`.")
        self._validate_metadata(target, resolved_step)
        manager = self._active_manager()
        return cast(TrainState, manager.restore(resolved_step, args=ocp.args.StandardRestore(target)))

    def _validate_metadata(self, target: TrainState, step: int) -> None:
        """Validate checkpoint schema, producer line, step, and target structure.

        Args:
            target: Restore template defining the expected state structure.
            step: Resolved checkpoint step.

        Raises:
            ValueError: If metadata is absent or incompatible with this runtime and target.
        """
        metadata = self._active_manager().metadata(step).custom_metadata.get("phijax")
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint does not contain a PhiJAX compatibility manifest.")
        if metadata.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Checkpoint schema `{metadata.get('schema_version')}` is incompatible with "
                f"schema `{_CHECKPOINT_SCHEMA_VERSION}`."
            )
        producer = metadata.get("phijax_version")
        if not isinstance(producer, str) or _version_line(producer) != _version_line(version("phijax")):
            raise ValueError(
                f"Checkpoint produced by PhiJAX `{producer}` cannot be restored by PhiJAX `{version('phijax')}`; "
                "checkpoints are compatible only within one major/minor release line."
            )
        if metadata.get("step") != step:
            manifest_step = metadata.get("step")
            raise ValueError(f"Checkpoint manifest step `{manifest_step}` does not match directory step `{step}`.")
        expected_identifier = _state_identifier(target)
        if metadata.get("state_identifier") != expected_identifier:
            raise ValueError("Checkpoint state structure does not match the supplied restore template.")

    def restore_weights(self, target: TrainState, step: int | None = None) -> TrainState:
        """Restore model weights while preserving fresh optimization and run state.

        Args:
            target: Fresh training state defining both restore structure and preserved non-model fields.
            step: Checkpoint step, or `None` for the latest committed step.

        Returns:
            `target` with only `model_state` replaced from the checkpoint.
        """
        restored = self.restore(target, step)
        return replace(target, model_state=restored.model_state)

    def restore_callback_states(self, step: int | None = None) -> dict[str, Mapping[str, Any]]:
        """Restore JSON-compatible callback states from checkpoint metadata.

        Args:
            step: Checkpoint step, or `None` for the latest committed step.

        Returns:
            Stable callback identifier mapping.

        Raises:
            FileNotFoundError: If no requested checkpoint exists.
            ValueError: If callback metadata is malformed.
        """
        resolved_step = self.latest_step if step is None else step
        if resolved_step is None:
            raise FileNotFoundError(f"No checkpoints exist below `{self.directory}`.")
        metadata = self._active_manager().metadata(resolved_step).custom_metadata.get("phijax")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("callbacks"), dict):
            raise ValueError("Checkpoint does not contain compatible callback state metadata.")
        callbacks = metadata["callbacks"]
        malformed = any(
            not isinstance(identifier, str) or not isinstance(state, dict) for identifier, state in callbacks.items()
        )
        if malformed:
            raise ValueError("Checkpoint callback state metadata is malformed.")
        return cast(dict[str, Mapping[str, Any]], callbacks)

    def wait_until_finished(self) -> None:
        """Block until all asynchronous checkpoint writes have committed."""
        if self._manager is not None:
            self._manager.wait_until_finished()

    def close(self) -> None:
        """Wait for pending writes and release Orbax resources idempotently."""
        if self._manager is None:
            return
        manager = self._manager
        self._manager = None
        manager.close()

    def __enter__(self) -> OrbaxCheckpointIO:
        """Enter a checkpoint-manager context.

        Returns:
            This checkpoint manager.
        """
        self.open()
        return self

    def __exit__(self, exception_type: Any, exception: Any, traceback: Any) -> None:
        """Close the manager when leaving a context.

        Args:
            exception_type: Active exception type, if any.
            exception: Active exception instance, if any.
            traceback: Active traceback, if any.
        """
        del exception_type, exception, traceback
        self.close()


def restore_checkpoint(
    target: TrainState,
    ckpt_path: str | Path | None,
    *,
    weights_only: bool = False,
    step: int | None = None,
) -> TrainState:
    """Optionally restore full training state or model weights from a checkpoint root.

    A missing path is the fresh-training case and returns `target` unchanged. Full restoration resumes model,
    optimizer, balancer, PRNG, precision, and step state. Weights-only restoration supports transfer learning while
    retaining every non-model field from `target`.

    Args:
        target: Fresh state providing the expected checkpoint structure.
        ckpt_path: Orbax checkpoint-root directory, or `None` for fresh training.
        weights_only: Whether to restore only `model_state` for transfer learning.
        step: Specific checkpoint step, or `None` for the latest committed step.

    Returns:
        Unchanged fresh state, fully resumed state, or weights-initialized fresh state.

    Raises:
        ValueError: If restoration options are supplied without a path, the path is empty, or `step` is invalid.
        FileNotFoundError: If the requested checkpoint does not exist.
    """
    if ckpt_path is None:
        if weights_only or step is not None:
            raise ValueError("`weights_only` and `ckpt_step` require a non-null `ckpt_path`.")
        return target
    if isinstance(ckpt_path, str) and not ckpt_path.strip():
        raise ValueError("`ckpt_path` must be a non-empty checkpoint-root path or `None`.")
    if step is not None and (isinstance(step, bool) or not isinstance(step, int) or step < 0):
        raise ValueError("`ckpt_step` must be a nonnegative integer or `None`.")
    with OrbaxCheckpointIO(ckpt_path, max_to_keep=None, enable_async_checkpointing=False) as checkpoint_io:
        if weights_only:
            return checkpoint_io.restore_weights(target, step)
        return checkpoint_io.restore(target, step)


__all__ = ["OrbaxCheckpointIO", "restore_checkpoint"]
