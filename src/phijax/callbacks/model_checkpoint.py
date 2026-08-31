import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol

import jax
import numpy as np

from phijax.callbacks.base import Callback, CallbackContext, TrainerContext

if TYPE_CHECKING:
    from phijax.training.trainer import Trainer

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
        callback_states: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> bool:
        """Submit a checkpoint to storage.

        Args:
            state: Functional state to persist.
            step: Unique checkpoint step.
            metrics: Optional host scalar metrics.
            force: Whether to bypass backend scheduling policies.
            callback_states: Optional JSON-compatible callback states.

        Returns:
            Whether the backend initiated a save.
        """
        ...

    @property
    def steps(self) -> tuple[int, ...]:
        """Return committed checkpoint steps in ascending order.

        Returns:
            Ordered checkpoint steps.
        """
        ...

    def checkpoint_path(self, step: int) -> Path | None:
        """Return a user-facing path for one checkpoint step.

        Args:
            step: Committed checkpoint step.

        Returns:
            Checkpoint path, or `None` when the backend has no filesystem path.
        """
        ...

    def delete(self, step: int) -> None:
        """Delete one committed checkpoint.

        Args:
            step: Checkpoint step to remove.
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

    def restore_callback_states(self, step: int | None = None) -> dict[str, Mapping[str, Any]]:
        """Restore callback state metadata.

        Args:
            step: Checkpoint step, or `None` for the latest checkpoint.

        Returns:
            Stable callback identifier mapping.
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
        monitor: Optional scalar metric used for top-k retention.
        mode: Whether smaller or larger monitored values are better.
        save_top_k: Number of best monitored checkpoints retained, `-1` for all, or `0` for none.
        best_model_path: Path of the best retained checkpoint, when available.
        best_model_score: Best retained monitored value, when available.
        last_model_path: Path of the terminal checkpoint, when available.
    """

    def __init__(
        self,
        checkpoint_io: CheckpointIO,
        *,
        every_n_steps: int | None = None,
        save_last: bool = True,
        save_on_exception: bool = False,
        monitor: str | None = None,
        mode: Literal["min", "max"] = "min",
        save_top_k: int | None = None,
    ) -> None:
        """Initialize checkpoint lifecycle policy.

        Args:
            checkpoint_io: Storage backend used for checkpoint operations.
            every_n_steps: Positive periodic save interval, or `None` to disable periodic saves.
            save_last: Whether to save the final state passed to :meth:`on_fit_end`.
            save_on_exception: Whether to save the last valid state from :meth:`on_exception`.
            monitor: Optional scalar metric used to rank checkpoints.
            mode: `"min"` or `"max"` monitored-metric optimization direction.
            save_top_k: Retained best checkpoint count. `None` selects one with monitoring and all without it.

        Raises:
            ValueError: If checkpoint scheduling or monitoring options are invalid.
        """
        if every_n_steps is not None and every_n_steps < 1:
            raise ValueError("`every_n_steps` must be positive or `None`.")
        if monitor is not None and (not isinstance(monitor, str) or not monitor.strip()):
            raise ValueError("`monitor` must be a non-empty metric name or `None`.")
        if mode not in {"min", "max"}:
            raise ValueError("`mode` must be `min` or `max`.")
        resolved_top_k = (1 if monitor is not None else -1) if save_top_k is None else save_top_k
        if isinstance(resolved_top_k, bool) or not isinstance(resolved_top_k, int) or resolved_top_k < -1:
            raise ValueError("`save_top_k` must be `-1` or a nonnegative integer.")
        if monitor is None and resolved_top_k not in {-1, 0}:
            raise ValueError("Finite `save_top_k` requires a monitored metric.")
        self.checkpoint_io = checkpoint_io
        self.every_n_steps = every_n_steps
        self.save_last = save_last
        self.save_on_exception = save_on_exception
        self.monitor = monitor
        self.mode = mode
        self.save_top_k = resolved_top_k
        self.best_model_path: Path | None = None
        self.best_model_score: float | None = None
        self.last_model_path: Path | None = None
        self._scores: dict[int, float] = {}
        self._last_saved_step: int | None = None
        self._trainer: Trainer | None = None

    def connect(self, trainer: Any) -> None:
        """Connect checkpoint persistence to Trainer callback state.

        Args:
            trainer: Owning Trainer.
        """
        self._trainer = trainer

    def setup(self) -> None:
        """Reset fit-local checkpoint scheduling state."""
        self.best_model_path = None
        self.best_model_score = None
        self.last_model_path = None
        self._scores = {}
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
        return False

    def on_train_metrics(self, context: TrainerContext) -> None:
        """Evaluate periodic checkpoint policy after all metrics are available.

        Args:
            context: Complete post-update metric context.
        """
        if (
            self.every_n_steps is not None
            and context.step > 0
            and context.step % self.every_n_steps == 0
            and context.step != self._last_saved_step
        ):
            self._save(context, save_as_last=False)

    def on_fit_end(self, context: TrainerContext) -> None:
        """Save the terminal state when `save_last` is enabled.

        Args:
            context: Final trainer state, step, and metrics.
        """
        if self.save_last and context.step != self._last_saved_step:
            self._save(context, save_as_last=True)
        elif self.save_last:
            self.last_model_path = self.checkpoint_io.checkpoint_path(context.step)

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
            self._save(context, save_as_last=True)

    def teardown(self) -> None:
        """Wait for asynchronous writes and release stage resources."""
        self.checkpoint_io.close()

    def close(self) -> None:
        """Wait for pending writes and release checkpoint backend resources idempotently."""
        self.checkpoint_io.close()

    def _save(self, context: TrainerContext, *, save_as_last: bool) -> None:
        """Transfer scalar metrics to the host and submit one complete checkpoint.

        Args:
            context: State, step, and metrics associated with the checkpoint.
            save_as_last: Whether the checkpoint is retained as the terminal state independently of top-k ranking.
        """
        if self.checkpoint_io.latest_step == context.step:
            self._last_saved_step = context.step
            return
        previous_state = self.state_dict()
        jax.block_until_ready(context.metrics)
        if self.monitor is not None and self.monitor in context.metrics:
            monitored = np.asarray(jax.device_get(context.metrics[self.monitor]))
            if monitored.size != 1:
                raise ValueError(
                    f"Monitored checkpoint metric `{self.monitor}` must be scalar, received shape {monitored.shape}."
                )
        metrics = _scalar_metrics(context.metrics)
        score = self._monitor_score(metrics)
        qualifies = self._qualifies(score)
        if not qualifies and not save_as_last:
            return
        displaced_step = self._displaced_step(score) if qualifies else None
        path = self.checkpoint_io.checkpoint_path(context.step)
        if qualifies and score is not None:
            self._scores[context.step] = score
            if displaced_step is not None:
                del self._scores[displaced_step]
            self._update_best()
        if save_as_last:
            self.last_model_path = path
        self._last_saved_step = context.step
        callback_states = self._trainer.callback_state_dict() if self._trainer is not None else context.callback_states
        saved = self.checkpoint_io.save(
            context.state,
            context.step,
            metrics,
            force=True,
            callback_states=callback_states,
        )
        if saved:
            if displaced_step is not None and displaced_step != context.step:
                self.checkpoint_io.delete(displaced_step)
            log.info(f"Checkpoint submitted at step {context.step}.")
        else:
            self.load_state_dict(previous_state)

    def _monitor_score(self, metrics: Mapping[str, float]) -> float | None:
        """Read and validate the configured monitored value.

        Args:
            metrics: Complete host scalar metric mapping.

        Returns:
            Monitored score, or `None` when monitoring is disabled.

        Raises:
            ValueError: If the configured monitored metric is unavailable.
        """
        if self.monitor is None:
            return None
        if self.monitor not in metrics:
            raise ValueError(f"Monitored checkpoint metric `{self.monitor}` is unavailable.")
        return metrics[self.monitor]

    def _qualifies(self, score: float | None) -> bool:
        """Return whether a candidate belongs in monitored retention.

        Args:
            score: Candidate monitored score, or `None` without monitoring.

        Returns:
            Whether the candidate should be saved by top-k policy.

        Raises:
            RuntimeError: If monitored retention is evaluated without a score.
        """
        if self.monitor is None:
            return self.save_top_k != 0
        if self.save_top_k == 0:
            return False
        if self.save_top_k == -1 or len(self._scores) < self.save_top_k:
            return True
        if score is None:
            raise RuntimeError("Monitored checkpoint retention requires a scalar score.")
        worst_score = max(self._scores.values()) if self.mode == "min" else min(self._scores.values())
        return score < worst_score if self.mode == "min" else score > worst_score

    def _displaced_step(self, score: float | None) -> int | None:
        """Select the deterministic worst checkpoint displaced by a candidate.

        Args:
            score: Qualifying candidate score.

        Returns:
            Existing checkpoint step to delete, or `None`.
        """
        del score
        if self.monitor is None or self.save_top_k in {-1, 0} or len(self._scores) < self.save_top_k:
            return None
        key = (lambda item: (item[1], item[0])) if self.mode == "min" else (lambda item: (-item[1], item[0]))
        return max(self._scores.items(), key=key)[0]

    def _update_best(self) -> None:
        """Refresh public best-checkpoint attributes from retained scores."""
        if not self._scores:
            self.best_model_path = None
            self.best_model_score = None
            return
        best = (
            min(self._scores.items(), key=lambda item: (item[1], item[0]))
            if self.mode == "min"
            else max(self._scores.items(), key=lambda item: (item[1], -item[0]))
        )
        self.best_model_path = self.checkpoint_io.checkpoint_path(best[0])
        self.best_model_score = best[1]

    def state_dict(self) -> Mapping[str, Any]:
        """Return checkpoint ranking and path state.

        Returns:
            JSON-compatible persistent callback state.
        """
        return {
            "scores": {str(step): score for step, score in self._scores.items()},
            "last_saved_step": self._last_saved_step,
            "best_model_path": None if self.best_model_path is None else str(self.best_model_path),
            "best_model_score": self.best_model_score,
            "last_model_path": None if self.last_model_path is None else str(self.last_model_path),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore checkpoint ranking and path state.

        Args:
            state: Persistent state returned by :meth:`state_dict`.

        Raises:
            ValueError: If the saved state is malformed.
        """
        required = {"scores", "last_saved_step", "best_model_path", "best_model_score", "last_model_path"}
        if set(state) != required or not isinstance(state["scores"], Mapping):
            raise ValueError("ModelCheckpoint state is malformed.")
        try:
            self._scores = {int(step): float(score) for step, score in state["scores"].items()}
        except (TypeError, ValueError) as error:
            raise ValueError("ModelCheckpoint scores are malformed.") from error
        last_saved_step = state["last_saved_step"]
        if last_saved_step is not None and not isinstance(last_saved_step, int):
            raise ValueError("ModelCheckpoint `last_saved_step` is malformed.")
        self._last_saved_step = last_saved_step
        best_path = state["best_model_path"]
        last_path = state["last_model_path"]
        self.best_model_path = None if best_path is None else Path(str(best_path))
        self.last_model_path = None if last_path is None else Path(str(last_path))
        best_score = state["best_model_score"]
        self.best_model_score = None if best_score is None else float(best_score)


def _scalar_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    """Convert scalar device metrics to host floats for checkpoint metadata.

    Args:
        metrics: Named scalar arrays or Python numbers.

    Returns:
        Host scalar metric mapping.
    """
    result: dict[str, float] = {}
    for name, value in metrics.items():
        array = np.asarray(jax.device_get(value))
        if array.size != 1:
            continue
        result[name] = float(array.reshape(()))
    return result


__all__ = ["CheckpointIO", "ModelCheckpoint"]
