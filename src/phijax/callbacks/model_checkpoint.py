import logging
from collections.abc import Mapping
from typing import Any, Protocol

import jax
import numpy as np

from phijax.callbacks.base import Callback, CallbackContext, TrainerContext

log = logging.getLogger(__name__)


class CheckpointIO(Protocol):
    """Describe the storage operations required by :class:`ModelCheckpoint`.

    Backends must allow `open()` after `close()`. Both methods must be idempotent so each Trainer task can own its
    resources independently.
    """

    def open(self) -> None:
        """Prepare checkpoint backend resources for a new fit stage idempotently."""
        ...

    @property
    def latest_step(self) -> int | None:
        """Return the latest committed checkpoint step.

        Returns:
            Latest checkpoint step, or `None` when storage is empty.
        """
        ...

    def save(
        self,
        state: Any,
        step: int,
        metrics: Mapping[str, float] | None = None,
        *,
        force: bool = False,
    ) -> bool:
        """Submit a checkpoint to storage.

        Args:
            state: Functional state to persist.
            step: Unique checkpoint step.
            metrics: Optional host scalar metrics.
            force: Whether to bypass backend scheduling policies.

        Returns:
            Whether the backend initiated a save.
        """
        ...

    def restore(self, target: Any, step: int | None = None) -> Any:
        """Restore a complete state.

        Args:
            target: State defining the expected restore structure.
            step: Checkpoint step, or `None` for the latest checkpoint.

        Returns:
            Restored complete state.
        """
        ...

    def restore_weights(self, target: Any, step: int | None = None) -> Any:
        """Restore model weights into a fresh state.

        Args:
            target: Fresh state whose non-model fields are preserved.
            step: Checkpoint step, or `None` for the latest checkpoint.

        Returns:
            State containing restored model weights.
        """
        ...

    def wait_until_finished(self) -> None:
        """Wait until pending checkpoint writes have committed."""
        ...

    def close(self) -> None:
        """Wait for pending writes and release checkpoint backend resources idempotently."""
        ...


class ModelCheckpoint(Callback):
    """Save functional training states at a configured interval and at fit completion.

    The callback owns checkpoint scheduling policy. Its `checkpoint_io` backend owns serialization and retention, so
    alternative storage implementations can be supplied without changing trainer orchestration.

    Attributes:
        checkpoint_io: Storage backend used for save and restore operations.
        every_n_steps: Periodic save interval, or `None` to disable periodic saves.
        save_last: Whether to save the terminal fit state.
        save_on_exception: Whether to save the last valid state when training is interrupted or fails.
    """

    def __init__(
        self,
        checkpoint_io: CheckpointIO,
        *,
        every_n_steps: int | None = None,
        save_last: bool = True,
        save_on_exception: bool = False,
    ) -> None:
        """Initialize checkpoint lifecycle policy.

        Args:
            checkpoint_io: Storage backend used for checkpoint operations.
            every_n_steps: Positive periodic save interval, or `None` to disable periodic saves.
            save_last: Whether to save the final state passed to :meth:`on_fit_end`.
            save_on_exception: Whether to save the last valid state from :meth:`on_exception`.

        Raises:
            ValueError: If `every_n_steps` is not positive or `None`.
        """
        if every_n_steps is not None and every_n_steps < 1:
            raise ValueError("`every_n_steps` must be positive or `None`.")
        self.checkpoint_io = checkpoint_io
        self.every_n_steps = every_n_steps
        self.save_last = save_last
        self.save_on_exception = save_on_exception
        self._last_saved_step: int | None = None

    def setup(self) -> None:
        """Reset fit-local checkpoint scheduling state."""
        self._last_saved_step = None

    def on_fit_start(self, context: TrainerContext) -> None:
        """Open checkpoint resources for the fit stage.

        Args:
            context: Initial trainer context.
        """
        del context
        self.checkpoint_io.open()

    def on_train_batch_end(self, context: TrainerContext) -> bool:
        """Save the state when the completed optimizer step matches the configured interval.

        Args:
            context: Post-update trainer state, step, and metrics.

        Returns:
            Always `False`; checkpointing never requests early termination.
        """
        if (
            self.every_n_steps is not None
            and context.step > 0
            and context.step % self.every_n_steps == 0
            and context.step != self._last_saved_step
        ):
            self._save(context)
        return False

    def on_fit_end(self, context: TrainerContext) -> None:
        """Save the terminal state when `save_last` is enabled.

        Args:
            context: Final trainer state, step, and metrics.
        """
        if self.save_last and context.step != self._last_saved_step:
            self._save(context)

    def on_exception(self, exception: BaseException, context: CallbackContext) -> None:
        """Optionally save the most recent valid state during exceptional shutdown.

        Args:
            exception: Exception or interruption terminating the fit call.
            context: Most recent internally consistent trainer context.
        """
        del exception
        if not isinstance(context, TrainerContext):
            return
        if self.save_on_exception and context.step != self._last_saved_step:
            self._save(context)

    def teardown(self) -> None:
        """Wait for asynchronous writes and release stage resources."""
        self.checkpoint_io.close()

    def close(self) -> None:
        """Wait for pending writes and release checkpoint backend resources idempotently."""
        self.checkpoint_io.close()

    def _save(self, context: TrainerContext) -> None:
        """Transfer scalar metrics to the host and submit one complete checkpoint.

        Args:
            context: State, step, and metrics associated with the checkpoint.
        """
        if self.checkpoint_io.latest_step == context.step:
            self._last_saved_step = context.step
            return
        jax.block_until_ready(context.metrics)
        saved = self.checkpoint_io.save(
            context.state,
            context.step,
            _scalar_metrics(context.metrics),
            force=True,
        )
        if saved:
            self._last_saved_step = context.step
            log.info(f"Checkpoint submitted at step {context.step}.")


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Convert scalar device metrics to host floats for checkpoint metadata.

    Args:
        metrics: Named scalar arrays or Python numbers.

    Returns:
        Host scalar metric mapping.

    Raises:
        ValueError: If a metric contains more than one value.
    """
    result: dict[str, float] = {}
    for name, value in metrics.items():
        array = np.asarray(jax.device_get(value))
        if array.size != 1:
            raise ValueError(f"Checkpoint metric `{name}` must be scalar, received shape {array.shape}.")
        result[name] = float(array.reshape(()))
    return result


__all__ = ["CheckpointIO", "ModelCheckpoint"]
