from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal

from phijax.data.pools import HostPool

type MonitorMode = Literal["min", "max"]


@dataclass(frozen=True, slots=True)
class TrainerContext:
    """Expose immutable trainer progress to callbacks.

    Attributes:
        state: Current functional training-state PyTree.
        step: Current training step as a Python integer.
        metrics: Most recent device or host metric mapping.
        module: Application module participating in the fit lifecycle, when available.
        is_global_zero: Whether the callback is running on the global rank-zero process.
        should_log: Whether the trainer logger cadence selects the current completed step.
        is_fit_end: Whether the context represents the terminal state of a fit pass.
        batch: Device-placed training batch during `on_train_batch_start`, or `None` for other fit hooks.
        has_logger: Whether the Trainer was configured with at least one experiment logger.
        callback_states: Host callback states included in a checkpoint request.
    """

    state: Any
    step: int
    metrics: Mapping[str, Any]
    module: Any | None = None
    is_global_zero: bool = True
    should_log: bool = False
    is_fit_end: bool = False
    batch: Any | None = None
    has_logger: bool = False
    callback_states: Mapping[str, Mapping[str, Any]] | None = None


@dataclass(frozen=True, slots=True)
class PredictionContext:
    """Expose one prediction lifecycle event to callbacks.

    Attributes:
        outputs: Prediction PyTree, or `None` before outputs are available.
        batch: Current prediction batch, or `None` outside batch-level hooks.
        batch_index: Zero-based prediction batch index, or `None` for task-level events.
        total_batches: Number of batches in the finite prediction source, or `None` when its length is unknown.
        metadata: Immutable run, sample, or output metadata supplied by the prediction entrypoint.
        pool: Immutable host pool used to reconstruct flat outputs, when exposed by the prediction source.
        is_global_zero: Whether the callback is running on the global rank-zero process.
    """

    outputs: Any | None
    batch_index: int | None
    metadata: Mapping[str, Any]
    batch: Mapping[str, Any] | None = None
    total_batches: int | None = None
    pool: HostPool | None = None
    is_global_zero: bool = True


@dataclass(frozen=True, slots=True)
class PostprocessingContext:
    """Expose postprocessing inputs or results to callbacks.

    Attributes:
        value: Current postprocessing input or result PyTree.
        metadata: Immutable run, sample, or artifact metadata supplied by the entrypoint.
    """

    value: Any
    metadata: Mapping[str, Any]


type CallbackContext = TrainerContext | PredictionContext | PostprocessingContext


class Callback:
    """Define optional host-side hooks shared by training, prediction, and postprocessing.

    Hooks execute in Python and must never be called from a transformed JAX function. Subclasses may retain host-side
    bookkeeping or write artifacts, but numerical model and optimizer changes belong in compiled functions.
    """

    def connect(self, trainer: Any) -> None:
        """Connect the callback to its owning Trainer before a task begins.

        Args:
            trainer: Host Trainer coordinating callback hooks.
        """
        del trainer
        return None

    def setup(self) -> None:
        """Prepare external callback resources before a task starts."""
        return None

    def on_fit_start(self, context: TrainerContext) -> None:
        """Handle the beginning of a fit call.

        Args:
            context: Immutable initial trainer context.
        """
        del context
        return None

    def on_train_batch_start(self, context: TrainerContext) -> None:
        """Handle the beginning of one training iteration.

        Args:
            context: Trainer context containing the device-placed batch before module transformations.
        """
        del context
        return None

    def on_train_batch_end(self, context: TrainerContext) -> bool:
        """Handle a completed training iteration.

        Args:
            context: Trainer context after the compiled update.

        Returns:
            `True` to request a clean early stop, otherwise `False`.
        """
        return False

    def training_metrics(self, context: TrainerContext) -> Mapping[str, Any]:
        """Contribute explicit scalar metrics after module batch-end processing.

        Args:
            context: Post-module trainer context for one completed optimizer step.

        Returns:
            Additional uniquely named metrics merged into logging and fit results.
        """
        del context
        return {}

    def on_train_metrics(self, context: TrainerContext) -> None:
        """Handle the complete metric mapping for one optimizer step.

        Args:
            context: Post-update context after module and callback metrics have been merged.
        """
        del context
        return None

    def on_fit_end(self, context: TrainerContext) -> None:
        """Handle the end of a successful or callback-stopped fit call.

        Args:
            context: Final trainer context.
        """
        del context
        return None

    def on_predict_start(self, context: PredictionContext) -> None:
        """Handle the beginning of a prediction call.

        Args:
            context: Initial prediction context.
        """
        del context
        return None

    def on_predict_epoch_start(self, context: PredictionContext) -> None:
        """Handle the beginning of the finite prediction pass.

        Args:
            context: Initial prediction-pass context.
        """
        del context
        return None

    def on_predict_batch_start(self, context: PredictionContext) -> None:
        """Handle the beginning of one prediction batch.

        Args:
            context: Prediction context containing the placed batch.
        """
        del context
        return None

    def on_predict_batch_end(self, context: PredictionContext) -> None:
        """Handle outputs from one prediction batch.

        Args:
            context: Prediction context containing batch outputs.
        """
        del context
        return None

    def on_predict_epoch_end(self, context: PredictionContext) -> None:
        """Handle completion of the finite prediction pass.

        Args:
            context: Prediction context containing assembled outputs when collection is enabled.
        """
        del context
        return None

    def on_predict_end(self, context: PredictionContext) -> None:
        """Handle completion of prediction and output assembly.

        Args:
            context: Final prediction context.
        """
        del context
        return None

    def on_postprocessing_start(self, context: PostprocessingContext) -> None:
        """Handle the beginning of postprocessing.

        Args:
            context: Initial postprocessing context.
        """
        del context
        return None

    def on_postprocessing_end(self, context: PostprocessingContext) -> None:
        """Handle a completed postprocessing result.

        Args:
            context: Final postprocessing context.
        """
        del context
        return None

    def on_exception(self, exception: BaseException, context: CallbackContext) -> None:
        """Handle an exception before its owning task re-raises it.

        Args:
            exception: Exception raised during a callback-enabled lifecycle.
            context: Most recent valid lifecycle context.
        """
        del exception, context
        return None

    def teardown(self) -> None:
        """Release callback resources after the owning task terminates."""
        return None

    def state_dict(self) -> Mapping[str, Any]:
        """Return JSON-compatible persistent callback state.

        Returns:
            Callback state stored with checkpoints, empty for stateless callbacks.
        """
        return {}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore persistent callback state.

        Args:
            state: State previously returned by :meth:`state_dict`.

        Raises:
            ValueError: If a stateless callback receives non-empty state.
        """
        if state:
            raise ValueError(f"{type(self).__name__} does not define persistent callback state.")


__all__ = [
    "Callback",
    "CallbackContext",
    "MonitorMode",
    "PostprocessingContext",
    "PredictionContext",
    "TrainerContext",
]
