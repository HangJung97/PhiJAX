from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from phijax.callbacks.base import PredictionContext


@dataclass(frozen=True, slots=True)
class PhiModuleContext:
    """Expose module-specific host-loop progress without optimizer or balancer state.

    Attributes:
        step: Current completed optimizer step as a Python integer.
        metrics: Most recent device or host metric mapping.
    """

    step: int
    metrics: Mapping[str, Any]


class _ModuleHooks:
    """Provide overridable host lifecycle hooks for :class:`phijax.core.BasePhiModule`."""

    def on_predict_start(self, model_state: Any, context: PredictionContext) -> None:
        """Handle the beginning of a prediction call.

        Args:
            model_state: Explicit model state after device placement.
            context: Initial prediction context.
        """
        del model_state, context
        return None

    def on_predict_epoch_start(self, model_state: Any, context: PredictionContext) -> None:
        """Handle the beginning of the finite prediction pass.

        Args:
            model_state: Explicit model state used for prediction.
            context: Initial prediction-pass context.
        """
        del model_state, context
        return None

    def on_predict_batch_start(self, model_state: Any, context: PredictionContext) -> None:
        """Handle the beginning of one prediction batch.

        Args:
            model_state: Explicit model state used for prediction.
            context: Prediction context containing the placed batch.
        """
        del model_state, context
        return None

    def on_predict_batch_end(self, model_state: Any, context: PredictionContext) -> None:
        """Handle outputs from one prediction batch.

        Args:
            model_state: Explicit model state used for prediction.
            context: Prediction context containing valid unpadded outputs.
        """
        del model_state, context
        return None

    def on_predict_epoch_end(self, model_state: Any, context: PredictionContext) -> None:
        """Handle completion of the finite prediction pass.

        Args:
            model_state: Explicit model state used for prediction.
            context: Prediction context containing assembled outputs when collection is enabled.
        """
        del model_state, context
        return None

    def on_predict_end(self, model_state: Any, context: PredictionContext) -> None:
        """Handle completion of a prediction call.

        Args:
            model_state: Explicit model state used for prediction.
            context: Final prediction context.
        """
        del model_state, context
        return None

    def setup(self) -> None:
        """Prepare application-specific host resources before a Trainer task."""
        return None

    def on_fit_start(self, model_state: Any, context: PhiModuleContext) -> Any:
        """Transform model state before the first training batch.

        Args:
            model_state: Explicit model state after restore and device placement.
            context: Initial module lifecycle context.

        Returns:
            Model state used by the first training iteration.
        """
        del context
        return model_state

    def on_train_batch_start(
        self,
        model_state: Any,
        batch: Any,
        context: PhiModuleContext,
    ) -> tuple[Any, Any]:
        """Transform model state or a batch before the compiled training step.

        Args:
            model_state: Current explicit model state.
            batch: Cast and device-placed training batch.
            context: Pre-update module lifecycle context.

        Returns:
            Model state and batch passed to the compiled training step.
        """
        del context
        return model_state, batch

    def on_train_batch_end(
        self,
        model_state: Any,
        context: PhiModuleContext,
    ) -> tuple[Any, Mapping[str, Any]]:
        """Transform model state or metrics after the compiled training step.

        Args:
            model_state: Updated explicit model state.
            context: Post-update context containing the compiled step metrics.

        Returns:
            Model state and metrics exposed to callbacks and loggers.
        """
        return model_state, context.metrics

    def on_fit_end(self, model_state: Any, context: PhiModuleContext) -> Any:
        """Transform terminal model state after successful or callback-stopped training.

        Args:
            model_state: Terminal explicit model state.
            context: Final module lifecycle context.

        Returns:
            Model state returned in the fit result and exposed to callbacks.
        """
        del context
        return model_state

    def on_exception(self, exception: BaseException, context: PhiModuleContext) -> None:
        """Handle a fit exception before the Trainer re-raises it.

        Args:
            exception: Exception raised during the module-enabled fit lifecycle.
            context: Most recent valid module lifecycle context.
        """
        del exception, context
        return None

    def teardown(self) -> None:
        """Release application-specific host resources after a Trainer task."""
        return None


__all__ = ["PhiModuleContext"]
