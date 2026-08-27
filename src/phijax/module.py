from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax

from phijax.callbacks.base import PredictionContext
from phijax.objectives import Objective, ResidualObjective
from phijax.types import ArrayMapping, ModelApply, ModelSummaryFunction, NamedBatches


@dataclass(frozen=True, slots=True)
class PhiModuleContext:
    """Expose module-specific host-loop progress without optimizer or balancer state.

    Attributes:
        step: Current completed optimizer step as a Python integer.
        metrics: Most recent device or host metric mapping.
    """

    step: int
    metrics: Mapping[str, Any]


class BasePhiModule(ABC):
    """Define the application-facing computation and lifecycle contract used by :class:`Trainer`.

    A module owns model application and objective behavior but never owns optimizer or loss-balancer logic. Mutable
    arrays remain in :class:`phijax.training.TrainState`, while training hooks return explicit model-state and batch
    replacements to the trainer. Prediction lifecycle hooks are observational; override :meth:`predict_step` to change
    outputs. Hooks execute on the Python host and must preserve JAX PyTree structures expected by compiled steps. The
    trainer invokes each callback hook before the matching module hook to follow Lightning's lifecycle convention.

    Attributes:
        name: Human-readable module name used by logging and configuration layers.
    """

    def __init__(self, *, name: str = "PINN") -> None:
        """Initialize the base module identity.

        Args:
            name: Non-empty human-readable module name.

        Raises:
            ValueError: If `name` is empty or whitespace-only.
        """
        if not name or not name.strip():
            raise ValueError("`name` must be non-empty.")
        self.name = name

    @property
    @abstractmethod
    def loss_names(self) -> Sequence[str]:
        """Return stable names for the unweighted losses produced by :meth:`training_step`.

        Returns:
            Ordered unique loss names used to initialize an independently selected balancer.
        """

    def __call__(self, model_state: Any, inputs: jax.Array) -> jax.Array:
        """Evaluate one model input.

        Args:
            model_state: Explicit model parameter PyTree.
            inputs: One model input point.

        Returns:
            Model output for `inputs`.
        """
        return self.forward(model_state, inputs)

    @abstractmethod
    def forward(self, model_state: Any, inputs: jax.Array) -> jax.Array:
        """Define the explicit-state forward computation.

        Args:
            model_state: Explicit model parameter PyTree.
            inputs: One model input point.

        Returns:
            Model output for `inputs`.
        """

    @abstractmethod
    def training_step(self, model_state: Any, batches: NamedBatches) -> Mapping[str, jax.Array]:
        """Evaluate named unweighted losses for one compiled training step.

        Args:
            model_state: Explicit differentiable model parameter PyTree.
            batches: Fixed-structure named objective batches.

        Returns:
            Scalar losses keyed by :attr:`loss_names`.
        """

    def format_training_metrics(
        self,
        total_loss: jax.Array,
        losses: Mapping[str, jax.Array],
        diagnostics: Mapping[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Format compiled training values for progress bars and experiment loggers.

        The combined loss becomes `train/loss`, each unweighted objective value becomes `train/loss/<name>`, and
        diagnostics become `train/<name>`. The host trainer subsequently sends every returned value to configured
        loggers, while the Rich progress callback selects the `train/loss` and `train/weight` namespaces.

        Args:
            total_loss: Scalar objective after external loss balancing.
            losses: Named unweighted scalar losses returned by :meth:`training_step`.
            diagnostics: Named scalar balancer, precision, or optimizer diagnostics.

        Returns:
            Stable flat metric mapping suitable for JIT output and host logging.
        """
        metrics = {"train/loss": total_loss}
        metrics.update({f"train/loss/{name}": value for name, value in losses.items()})
        metrics.update({f"train/{name}": value for name, value in diagnostics.items()})
        return metrics

    def residual_stream(self, name: str, model_state: Any, batches: NamedBatches) -> jax.Array:
        """Evaluate one residual stream for an adaptive external balancer.

        Args:
            name: One entry from :attr:`loss_names`.
            model_state: Explicit differentiable model parameter PyTree.
            batches: Fixed-structure named residual batches.

        Returns:
            Raw residual or model-output stream.

        Raises:
            NotImplementedError: If the module does not expose residual streams.
        """
        del name, model_state, batches
        raise NotImplementedError(f"{type(self).__name__} does not expose residual streams.")

    def summarize_model(
        self,
        model_state: Any,
        *,
        max_depth: int = -1,
        console_width: int = 120,
        compute_flops: bool = False,
        compute_vjp_flops: bool = False,
    ) -> str | None:
        """Render the underlying network architecture when the module supports summaries.

        Args:
            model_state: Explicit model parameter and variable state.
            max_depth: Maximum displayed module depth, or `-1` for every level.
            console_width: Positive Rich console width in characters.
            compute_flops: Whether to estimate forward-pass floating-point operations.
            compute_vjp_flops: Whether to estimate reverse-pass floating-point operations.

        Returns:
            Rendered model summary, or `None` when no summary provider is configured.
        """
        del model_state, max_depth, console_width, compute_flops, compute_vjp_flops
        return None

    def predict_step(self, model_state: Any, batch: ArrayMapping) -> jax.Array:
        """Evaluate one prediction batch.

        Args:
            model_state: Explicit model parameter PyTree.
            batch: Prediction batch containing model coordinates under `inputs`.

        Returns:
            Batched model outputs.
        """
        return jax.vmap(self, in_axes=(None, 0))(model_state, batch["inputs"])

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
        """Prepare application-specific host resources before a trainer task."""
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
        """Handle a fit exception before the trainer re-raises it.

        Args:
            exception: Exception raised during the module-enabled fit lifecycle.
            context: Most recent valid module lifecycle context.
        """
        del exception, context
        return None

    def teardown(self) -> None:
        """Release application-specific host resources after a fit call."""
        return None


class PhiModule(BasePhiModule):
    """Bind a generic explicit-state model application callable to a configured objective.

    Attributes:
        model_apply: Pure explicit-state model application callable.
        objective: Objective producing unweighted scalar losses and optional residual streams.
        name: Human-readable module name.
    """

    def __init__(
        self,
        model_apply: ModelApply,
        objective: Objective,
        *,
        name: str = "PINN",
        model_summary: ModelSummaryFunction | None = None,
    ) -> None:
        """Initialize the configuration-driven module.

        Args:
            model_apply: Pure callable mapping explicit model state and one input point to model outputs.
            objective: Objective producing named unweighted scalar losses.
            name: Non-empty human-readable module name.
            model_summary: Optional callable rendering the explicit model state and network architecture.

        Raises:
            TypeError: If `model_apply` or a configured `model_summary` is not callable.
            ValueError: If `name` is empty or the objective exposes invalid loss names.
        """
        super().__init__(name=name)
        if not callable(model_apply):
            raise TypeError("`model_apply` must be callable.")
        if model_summary is not None and not callable(model_summary):
            raise TypeError("`model_summary` must be callable or `None`.")
        resolved_names = tuple(objective.loss_names)
        if (
            not resolved_names
            or any(not isinstance(loss_name, str) or not loss_name.strip() for loss_name in resolved_names)
            or len(set(resolved_names)) != len(resolved_names)
        ):
            raise ValueError("The objective must expose unique non-empty loss names.")
        self.model_apply = model_apply
        self.objective = objective
        self._loss_names = resolved_names
        self._model_summary = model_summary

    @property
    def loss_names(self) -> tuple[str, ...]:
        """Return stable objective loss names.

        Returns:
            Ordered unique objective loss names.
        """
        return self._loss_names

    def forward(self, model_state: Any, inputs: jax.Array) -> jax.Array:
        """Evaluate one model input with the configured application callable.

        Args:
            model_state: Explicit model parameter PyTree.
            inputs: One model input point.

        Returns:
            Model output for `inputs`.
        """
        return self.model_apply(model_state, inputs)

    def summarize_model(
        self,
        model_state: Any,
        *,
        max_depth: int = -1,
        console_width: int = 120,
        compute_flops: bool = False,
        compute_vjp_flops: bool = False,
    ) -> str | None:
        """Render the configured model summary.

        Args:
            model_state: Explicit model parameter and variable state.
            max_depth: Maximum displayed module depth, or `-1` for every level.
            console_width: Positive Rich console width in characters.
            compute_flops: Whether to estimate forward-pass floating-point operations.
            compute_vjp_flops: Whether to estimate reverse-pass floating-point operations.

        Returns:
            Rendered model summary, or `None` when no summary provider was supplied.
        """
        if self._model_summary is None:
            return None
        return self._model_summary(
            model_state,
            max_depth=max_depth,
            console_width=console_width,
            compute_flops=compute_flops,
            compute_vjp_flops=compute_vjp_flops,
        )

    def training_step(self, model_state: Any, batches: NamedBatches) -> Mapping[str, jax.Array]:
        """Evaluate the configured objective's named unweighted losses.

        Args:
            model_state: Explicit differentiable model parameter PyTree.
            batches: Fixed-structure named objective batches.

        Returns:
            Scalar objective losses keyed by :attr:`loss_names`.
        """
        return self.objective.losses(self, model_state, batches)

    def residual_stream(self, name: str, model_state: Any, batches: NamedBatches) -> jax.Array:
        """Evaluate one configured objective residual stream.

        Args:
            name: One entry from :attr:`loss_names`.
            model_state: Explicit differentiable model parameter PyTree.
            batches: Fixed-structure named residual batches.

        Returns:
            Raw residual or model-output stream.

        Raises:
            TypeError: If the configured objective does not support residual streams.
        """
        if not isinstance(self.objective, ResidualObjective):
            raise TypeError("The configured objective does not implement `ResidualObjective`.")
        return self.objective.residual_stream(name, self, model_state, batches)


__all__ = ["BasePhiModule", "PhiModule", "PhiModuleContext"]
