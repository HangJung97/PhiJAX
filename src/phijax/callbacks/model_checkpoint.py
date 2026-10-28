import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import jax
import numpy as np

from phijax.callbacks.base import Callback, CallbackContext, MonitorMode, TrainerContext

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

        Deleting a step that is not stored must be a no-op.

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

    The callback owns checkpoint scheduling and retention, following Lightning's `ModelCheckpoint`. Its
    `checkpoint_io` backend only stores, restores, and deletes checkpoints, so alternative storage implementations can
    be supplied without changing trainer orchestration.

    Retention:

    - With `monitor`, the callback keeps the best `save_top_k` checkpoints ranked by that metric. `step` is always
      available as a monitor, so `monitor="step", mode="max"` keeps the latest `save_top_k` checkpoints.
    - Without `monitor`, `save_top_k=1` keeps only the latest checkpoint, `-1` keeps every checkpoint, and `0` skips
      periodic saves.
    - The terminal checkpoint written by `save_last` or `save_on_exception` is never removed by ranking. The next
      terminal checkpoint replaces it.

    A checkpoint leaves storage only after the next checkpoint write has committed, so an interrupted asynchronous
    write never removes the only valid checkpoint.

    Attributes:
        best_model_path: Path of the best retained checkpoint, or of the latest one without `monitor`.
        best_model_score: Best retained monitored value, or `None` without `monitor`.
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
        mode: MonitorMode = "min",
        save_top_k: int = 1,
    ) -> None:
        """Initialize checkpoint lifecycle policy.

        Args:
            checkpoint_io: Storage backend used for checkpoint operations.
            every_n_steps: Positive periodic save interval, or `None` to disable periodic saves.
            save_last: Whether to save the final state passed to :meth:`on_fit_end`.
            save_on_exception: Whether to save the last valid state from :meth:`on_exception`.
            monitor: Optional scalar metric used to rank checkpoints. `step` ranks checkpoints by training step.
            mode: `"min"` or `"max"` monitored-metric optimization direction.
            save_top_k: Number of ranked checkpoints to keep. `-1` keeps all and `0` disables ranked saves. Without
                `monitor`, only `-1`, `0`, and `1` are valid, and `1` keeps the latest checkpoint.

        Raises:
            ValueError: If checkpoint scheduling or monitoring options are invalid.
        """
        if every_n_steps is not None and every_n_steps < 1:
            raise ValueError("`every_n_steps` must be positive or `None`.")
        if monitor is not None and (not isinstance(monitor, str) or not monitor.strip()):
            raise ValueError("`monitor` must be a non-empty metric name or `None`.")
        if mode not in {"min", "max"}:
            raise ValueError("`mode` must be `min` or `max`.")
        if isinstance(save_top_k, bool) or not isinstance(save_top_k, int) or save_top_k < -1:
            raise ValueError("`save_top_k` must be `-1` or a nonnegative integer.")
        if monitor is None and save_top_k not in {-1, 0, 1}:
            raise ValueError(
                "`save_top_k` must be `-1`, `0`, or `1` without `monitor`. "
                'Use `monitor="step", mode="max"` to keep the latest checkpoints.'
            )
        self.checkpoint_io = checkpoint_io
        self.every_n_steps = every_n_steps
        self.save_last = save_last
        self.save_on_exception = save_on_exception
        self.monitor = monitor
        self.mode: MonitorMode = mode
        self.save_top_k = save_top_k
        self.best_model_path: Path | None = None
        self.best_model_score: float | None = None
        self.last_model_path: Path | None = None
        self._scores: dict[int, float] = {}
        self._last_saved_step: int | None = None
        self._last_step: int | None = None
        self._pending_deletions: list[int] = []
        self._trainer: Trainer | None = None

    @property
    def _rank_mode(self) -> MonitorMode:
        """Return the ranking direction used for retention.

        Returns:
            `mode` with a monitor, otherwise `"max"` so the latest step ranks best.
        """
        return self.mode if self.monitor is not None else "max"

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
        self._last_step = None
        self._pending_deletions = []

    def on_fit_start(self, context: TrainerContext) -> None:
        """Open checkpoint resources and drop tracked steps that are no longer stored.

        Args:
            context: Initial trainer context.
        """
        del context
        self.checkpoint_io.open()
        if not self._scores and self._last_step is None:
            return
        # Restored callback state can reference steps that were deleted outside this callback.
        stored = set(self.checkpoint_io.steps)
        self._scores = {step: score for step, score in self._scores.items() if step in stored}
        self._update_best()
        if self._last_step is not None and self._last_step not in stored:
            self._last_step = None
            self.last_model_path = None

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
            # The final step was already saved periodically; it becomes the terminal checkpoint.
            previous_last_step = self._last_step
            self._last_step = context.step
            self.last_model_path = self.checkpoint_io.checkpoint_path(context.step)
            self._schedule_deletions(previous_last_step)

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
        """Wait for asynchronous writes, remove displaced checkpoints, and release stage resources."""
        self._flush_deletions()
        self.checkpoint_io.close()

    def close(self) -> None:
        """Wait for pending writes and release checkpoint backend resources idempotently."""
        self._flush_deletions()
        self.checkpoint_io.close()

    def _save(self, context: TrainerContext, *, save_as_last: bool) -> None:
        """Transfer scalar metrics to the host and submit one complete checkpoint.

        Args:
            context: State, step, and metrics associated with the checkpoint.
            save_as_last: Whether the checkpoint becomes the terminal checkpoint, independently of ranking.
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
        score = self._monitor_score(metrics, context.step)
        qualifies = self._qualifies(score)
        if not qualifies and not save_as_last:
            return
        displaced_step = self._displaced_step() if qualifies else None
        if qualifies:
            self._scores[context.step] = score
            if displaced_step is not None:
                del self._scores[displaced_step]
            self._update_best()
        previous_last_step = self._last_step
        if save_as_last:
            self._last_step = context.step
            self.last_model_path = self.checkpoint_io.checkpoint_path(context.step)
        self._last_saved_step = context.step
        # Deletions queued by the previous save run only after that write has committed.
        self._flush_deletions()
        callback_states = self._trainer.callback_state_dict() if self._trainer is not None else context.callback_states
        saved = self.checkpoint_io.save(
            context.state,
            context.step,
            metrics,
            force=True,
            callback_states=callback_states,
        )
        if not saved:
            self.load_state_dict(previous_state)
            return
        self._schedule_deletions(displaced_step, previous_last_step if save_as_last else None)
        log.info(f"Checkpoint submitted at step {context.step}.")

    def _schedule_deletions(self, *steps: int | None) -> None:
        """Queue checkpoints that are neither ranked nor terminal for deletion after the next commit.

        Args:
            *steps: Candidate checkpoint steps, where `None` entries are ignored.
        """
        retained = {self._last_step, self._last_saved_step, *self._scores}
        for step in steps:
            if step is not None and step not in retained and step not in self._pending_deletions:
                self._pending_deletions.append(step)

    def _flush_deletions(self) -> None:
        """Delete queued checkpoints once every pending checkpoint write has committed."""
        if not self._pending_deletions:
            return
        self.checkpoint_io.wait_until_finished()
        for step in self._pending_deletions:
            self.checkpoint_io.delete(step)
        self._pending_deletions = []

    def _monitor_score(self, metrics: Mapping[str, float], step: int) -> float:
        """Read the value that ranks a checkpoint.

        Args:
            metrics: Complete host scalar metric mapping.
            step: Training step of the candidate checkpoint.

        Returns:
            Monitored value, or `step` when monitoring is disabled.

        Raises:
            ValueError: If the configured monitored metric is unavailable.
        """
        if self.monitor is None:
            return float(step)
        # As in Lightning, `step` is always a monitor candidate; a logged metric with the same name takes precedence.
        candidates = {"step": float(step), **metrics}
        if self.monitor not in candidates:
            raise ValueError(f"Monitored checkpoint metric `{self.monitor}` is unavailable.")
        return candidates[self.monitor]

    def _qualifies(self, score: float) -> bool:
        """Return whether a candidate belongs in the ranked checkpoints.

        Args:
            score: Candidate ranking value.

        Returns:
            Whether the candidate should be saved by the top-k policy.
        """
        if self.save_top_k == 0:
            return False
        if self.save_top_k == -1 or len(self._scores) < self.save_top_k:
            return True
        worst_score = max(self._scores.values()) if self._rank_mode == "min" else min(self._scores.values())
        return score < worst_score if self._rank_mode == "min" else score > worst_score

    def _displaced_step(self) -> int | None:
        """Select the deterministic worst ranked checkpoint displaced by a qualifying candidate.

        Returns:
            Ranked checkpoint step to remove from the ranking, or `None` while the ranking has room.
        """
        if self.save_top_k in {-1, 0} or len(self._scores) < self.save_top_k:
            return None
        if self._rank_mode == "min":
            return max(self._scores.items(), key=lambda item: (item[1], item[0]))[0]
        return max(self._scores.items(), key=lambda item: (-item[1], item[0]))[0]

    def _update_best(self) -> None:
        """Refresh public best-checkpoint attributes from ranked scores."""
        if not self._scores:
            self.best_model_path = None
            self.best_model_score = None
            return
        best = (
            min(self._scores.items(), key=lambda item: (item[1], item[0]))
            if self._rank_mode == "min"
            else max(self._scores.items(), key=lambda item: (item[1], -item[0]))
        )
        self.best_model_path = self.checkpoint_io.checkpoint_path(best[0])
        self.best_model_score = best[1] if self.monitor is not None else None

    def state_dict(self) -> Mapping[str, Any]:
        """Return checkpoint ranking and path state.

        Returns:
            JSON-compatible persistent callback state.
        """
        return {
            "scores": {str(step): score for step, score in self._scores.items()},
            "last_saved_step": self._last_saved_step,
            "last_step": self._last_step,
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
        # PhiJAX 0.2.0b5 and earlier did not store `last_step`.
        if not required <= set(state) <= required | {"last_step"} or not isinstance(state["scores"], Mapping):
            raise ValueError("ModelCheckpoint state is malformed.")
        try:
            self._scores = {int(step): float(score) for step, score in state["scores"].items()}
        except (TypeError, ValueError) as error:
            raise ValueError("ModelCheckpoint scores are malformed.") from error
        last_saved_step = state["last_saved_step"]
        last_step = state.get("last_step")
        for name, value in (("last_saved_step", last_saved_step), ("last_step", last_step)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
                raise ValueError(f"ModelCheckpoint `{name}` is malformed.")
        self._last_saved_step = last_saved_step
        self._last_step = last_step
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
