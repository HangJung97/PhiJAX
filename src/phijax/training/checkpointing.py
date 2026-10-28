from __future__ import annotations

import json
import warnings
from collections.abc import Mapping
from dataclasses import fields, replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any, cast

import jax
import numpy as np
import orbax.checkpoint as ocp

from phijax.training.state import TrainState
from phijax.types import InputStatistics

_CHECKPOINT_SCHEMA_VERSION = 3
# Schema 2 (PhiJAX 0.2.0b5 and earlier) predates `TrainState.input_statistics` and is still restorable.
_LEGACY_SCHEMA_VERSION = 2


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


def _with_statistics_layout(target: TrainState, layout: Mapping[str, Any] | None) -> TrainState:
    """Give a restore target the input-statistics structure stored in a checkpoint.

    Orbax requires the restore target to match the stored tree, so the statistics layout comes from the checkpoint
    manifest rather than from the template.

    Args:
        target: Restore template.
        layout: Stored statistics layout from the checkpoint manifest, or `None`.

    Returns:
        `target` with statistics arrays or abstract placeholders matching the checkpoint.
    """
    if layout is None:
        return replace(target, input_statistics=None)
    shape, dtype = tuple(layout["shape"]), np.dtype(layout["dtype"])
    template = target.input_statistics
    if template is not None and tuple(template.mean.shape) == shape and np.dtype(template.mean.dtype) == dtype:
        return target
    # Place restored statistics like the other replicated scalar state.
    sharding = getattr(target.step, "sharding", None)
    placeholder = jax.ShapeDtypeStruct(shape, dtype, sharding=sharding)
    return replace(target, input_statistics=InputStatistics(mean=placeholder, std=placeholder))


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
    statistics = state.input_statistics
    # Restore targets read this layout so they can match the stored tree without relying on the template.
    layout = None
    if statistics is not None:
        layout = {"shape": [int(size) for size in statistics.mean.shape], "dtype": str(statistics.mean.dtype)}
    return {
        "phijax": {
            "schema_version": _CHECKPOINT_SCHEMA_VERSION,
            "phijax_version": version("phijax"),
            "step": step,
            "state_identifier": _state_identifier(state),
            "input_statistics": layout,
            "callbacks": dict(callback_states or {}),
        }
    }


class OrbaxCheckpointIO:
    """Manage lazily opened asynchronous full-state checkpoints with Orbax.

    The backend never deletes checkpoints on its own. :class:`~phijax.callbacks.ModelCheckpoint` owns retention through
    `save_top_k`, as Lightning's checkpoint callback does.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        enable_async_checkpointing: bool = True,
    ) -> None:
        """Initialize an Orbax checkpoint manager.

        Args:
            directory: Checkpoint-root directory.
            enable_async_checkpointing: Whether saves may finish in a background thread.
        """
        self.directory = Path(directory).expanduser().resolve()
        self._options = ocp.CheckpointManagerOptions(enable_async_checkpointing=enable_async_checkpointing)
        self._manager: ocp.CheckpointManager | None = None

    def open(self) -> None:
        """Open the Orbax manager when it is not already active."""
        if self._manager is None:
            # Registering the handler of the single saved item lets metadata reads describe it. Without it, Orbax logs
            # that item "default" could not be restored on every `metadata()` call.
            self._manager = ocp.CheckpointManager(
                self.directory,
                options=self._options,
                item_handlers=ocp.StandardCheckpointHandler(),
            )

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

        Deleting a step that is not stored is a no-op.

        Args:
            step: Checkpoint step to remove.
        """
        manager = self._active_manager()
        if step not in manager.all_steps():
            return
        manager.delete(step)
        manager.wait_until_finished()

    def _resolve_step(self, step: int | None) -> int:
        """Resolve an explicit or latest committed checkpoint step.

        Args:
            step: Checkpoint step, or `None` for the latest committed step.

        Returns:
            Committed checkpoint step.

        Raises:
            FileNotFoundError: If no checkpoint exists.
        """
        resolved_step = self.latest_step if step is None else step
        if resolved_step is None:
            raise FileNotFoundError(f"No checkpoints exist below `{self.directory}`.")
        return resolved_step

    def restore(self, target: TrainState, step: int | None = None) -> TrainState:
        """Restore a complete state for exact training resumption.

        The restored `input_statistics` come from the checkpoint, whatever the template holds. Checkpoints written
        before PhiJAX stored input statistics (schema 2) keep the template's statistics and emit a warning.

        Args:
            target: Abstract or concrete state defining the expected structure and sharding.
            step: Checkpoint step, or `None` for the latest committed step.

        Returns:
            Restored model, optimizer, balancer, PRNG, step, precision, and input-statistics state.

        Raises:
            FileNotFoundError: If no requested checkpoint exists.
        """
        resolved_step = self._resolve_step(step)
        manifest = self._validate_metadata(target, resolved_step)
        manager = self._active_manager()
        if manifest["schema_version"] == _LEGACY_SCHEMA_VERSION:
            warnings.warn(
                f"Checkpoint step {resolved_step} was written before PhiJAX stored input statistics. The restored "
                "state keeps the input statistics of the restore template.",
                UserWarning,
                stacklevel=2,
            )
            # A schema-2 tree has no `input_statistics` node, so restore every other field by name.
            legacy_target = {
                field.name: getattr(target, field.name)
                for field in fields(TrainState)
                if field.name != "input_statistics"
            }
            restored = manager.restore(resolved_step, args=ocp.args.StandardRestore(legacy_target))
            return TrainState(**restored, input_statistics=target.input_statistics)
        restore_target = _with_statistics_layout(target, manifest["input_statistics"])
        return cast(TrainState, manager.restore(resolved_step, args=ocp.args.StandardRestore(restore_target)))

    def _read_manifest(self, step: int) -> dict[str, Any]:
        """Read and validate the checkpoint schema, producer line, step, and statistics layout.

        Args:
            step: Resolved checkpoint step.

        Returns:
            Validated PhiJAX checkpoint manifest.

        Raises:
            ValueError: If the manifest is absent, malformed, or incompatible with this runtime.
        """
        metadata = self._active_manager().metadata(step).custom_metadata.get("phijax")
        if not isinstance(metadata, dict):
            raise ValueError("Checkpoint does not contain a PhiJAX compatibility manifest.")
        schema_version = metadata.get("schema_version")
        if schema_version not in {_LEGACY_SCHEMA_VERSION, _CHECKPOINT_SCHEMA_VERSION}:
            raise ValueError(
                f"Checkpoint schema `{schema_version}` is incompatible with schemas "
                f"`{_LEGACY_SCHEMA_VERSION}` and `{_CHECKPOINT_SCHEMA_VERSION}`."
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
        if schema_version == _CHECKPOINT_SCHEMA_VERSION:
            layout = metadata.get("input_statistics", ...)
            valid_layout = layout is None or (
                isinstance(layout, dict)
                and isinstance(layout.get("shape"), list)
                and len(layout["shape"]) == 1
                and all(isinstance(size, int) and size > 0 for size in layout["shape"])
                and isinstance(layout.get("dtype"), str)
            )
            if not valid_layout:
                raise ValueError("Checkpoint input-statistics layout is missing or malformed.")
        return metadata

    def _validate_metadata(self, target: TrainState, step: int) -> dict[str, Any]:
        """Validate the checkpoint manifest and the restore target structure.

        Args:
            target: Restore template defining the expected state structure.
            step: Resolved checkpoint step.

        Returns:
            Validated PhiJAX checkpoint manifest.

        Raises:
            ValueError: If metadata is absent or incompatible with this runtime and target.
        """
        manifest = self._read_manifest(step)
        if manifest["schema_version"] == _LEGACY_SCHEMA_VERSION:
            # `None` adds no leaves, so a schema-2 identifier matches the template without statistics.
            restore_target = replace(target, input_statistics=None)
        else:
            restore_target = _with_statistics_layout(target, manifest["input_statistics"])
        if manifest.get("state_identifier") != _state_identifier(restore_target):
            raise ValueError("Checkpoint state structure does not match the supplied restore template.")
        return manifest

    def restore_input_statistics(self, step: int | None = None) -> InputStatistics | None:
        """Read only the stored input statistics as host NumPy arrays.

        Use this to give a transfer-learning DataModule the source model's statistics, or a per-axis mix, from
        :meth:`~phijax.data.PhiDataModule.input_statistics`.

        Args:
            step: Checkpoint step, or `None` for the latest committed step.

        Returns:
            Stored `float32` statistics, or `None` when the checkpoint stores none or predates stored statistics.

        Raises:
            FileNotFoundError: If no requested checkpoint exists.
            ValueError: If the checkpoint manifest is absent or incompatible with this runtime.
        """
        resolved_step = self._resolve_step(step)
        layout = self._read_manifest(resolved_step).get("input_statistics")
        if layout is None:
            return None
        shape, dtype = tuple(layout["shape"]), np.dtype(layout["dtype"])
        item = {"input_statistics": {"mean": np.zeros(shape, dtype), "std": np.zeros(shape, dtype)}}
        restore_args = jax.tree.map(lambda _: ocp.RestoreArgs(restore_type=np.ndarray), item)
        restored = ocp.PyTreeCheckpointer().restore(
            # `CheckpointManager` stores its single unnamed item under `default`.
            self.checkpoint_path(resolved_step) / "default",
            args=ocp.args.PyTreeRestore(item, restore_args=restore_args, partial_restore=True),
        )
        statistics = restored["input_statistics"]
        return InputStatistics(mean=statistics["mean"], std=statistics["std"])

    def restore_weights(self, target: TrainState, step: int | None = None) -> TrainState:
        """Restore model weights while preserving fresh optimization and run state.

        Args:
            target: Fresh training state defining both restore structure and preserved non-model fields.
            step: Checkpoint step, or `None` for the latest committed step.

        Returns:
            `target` with only `model_state` replaced from the checkpoint. The target's input statistics are kept.
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
        resolved_step = self._resolve_step(step)
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
    with OrbaxCheckpointIO(ckpt_path, enable_async_checkpointing=False) as checkpoint_io:
        if weights_only:
            return checkpoint_io.restore_weights(target, step)
        return checkpoint_io.restore(target, step)


def load_input_statistics(ckpt_path: str | Path, *, step: int | None = None) -> InputStatistics | None:
    """Read the input statistics stored in a checkpoint without restoring the training state.

    Weights-only restoration (transfer learning) uses the current DataModule's statistics. To start from the source
    model's statistics instead, or to mix them per axis, return this function's result from
    :meth:`~phijax.data.PhiDataModule.input_statistics`.

    Args:
        ckpt_path: Orbax checkpoint-root directory.
        step: Checkpoint step, or `None` for the latest committed step.

    Returns:
        Stored statistics as host `float32` NumPy arrays, or `None` when the checkpoint stores none or was written
        before PhiJAX stored input statistics.

    Raises:
        FileNotFoundError: If the requested checkpoint does not exist.
        ValueError: If the checkpoint manifest is absent or incompatible with this runtime.
    """
    with OrbaxCheckpointIO(ckpt_path, enable_async_checkpointing=False) as checkpoint_io:
        return checkpoint_io.restore_input_statistics(step)


__all__ = ["OrbaxCheckpointIO", "load_input_statistics", "restore_checkpoint"]
