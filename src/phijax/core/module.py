from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from copy import copy
from typing import TYPE_CHECKING, Any, Self

import jax
import numpy as np

from phijax.core.hooks import PhiModuleContext, _ModuleHooks
from phijax.metrics import _ACTIVE_MODULE_METRICS, TrainingOutput, _metric_is_scalar
from phijax.objectives import Objective, ResidualObjective
from phijax.types import (
    ArrayMapping,
    InputPeriods,
    InputStatistics,
    ModelApply,
    ModelSummaryFunction,
    NamedBatches,
    ResolvedInputPeriods,
    as_input_periods,
)

if TYPE_CHECKING:
    from phijax.models.contracts import InitializedModel, ModelFactory
    from phijax.training.precision import PrecisionPolicy


class BasePhiModule(_ModuleHooks, ABC):
    """Define the application-facing computation and lifecycle contract used by :class:`Trainer`.

    A module owns model application and objective behavior but never owns optimizer or loss-balancer logic. Mutable
    arrays remain in :class:`phijax.training.TrainState`, while training hooks return explicit model-state and batch
    replacements to the trainer. Prediction lifecycle hooks are observational; override :meth:`predict_step` to change
    outputs. Hooks execute on the Python host and must preserve JAX PyTree structures expected by compiled steps. The
    trainer invokes each callback hook before the matching module hook to follow Lightning's lifecycle convention.
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

    @property
    def batch_keys(self) -> Sequence[str]:
        """Return batch names required by the module's training computation.

        Advanced modules used with :meth:`phijax.Trainer.fit_state` may leave this empty when their explicit
        :class:`phijax.TrainingPlan` already defines the batch routing.

        Returns:
            Ordered unique DataModule batch names, or an empty sequence when they cannot be inferred.
        """
        return ()

    @property
    def input_statistics(self) -> InputStatistics | None:
        """Return the input normalization statistics applied by the bound model.

        Custom modules own their normalization, so the default reports none. The Trainer stores the reported
        statistics in :class:`phijax.training.TrainState`.

        Returns:
            Bound statistics as host arrays, or `None`.
        """
        return None

    @property
    def input_periods(self) -> ResolvedInputPeriods | None:
        """Return the physical input periods applied by the bound model.

        Custom modules own their input handling, so the default reports none. The Trainer stores the reported periods
        in :class:`phijax.training.TrainState`.

        Returns:
            Sorted `(axis, period)` pairs, or `None`.
        """
        return None

    def bind_inputs(self, statistics: InputStatistics | None, periods: InputPeriods | None) -> Self:
        """Return a module whose model applies `statistics` and `periods`.

        The Trainer calls this after restoring a checkpoint so the module uses the restored inputs. Custom modules own
        their input handling, so the default returns this module unchanged.

        Args:
            statistics: Normalization statistics to apply, or `None`.
            periods: Physical input periods to apply, or `None`.

        Returns:
            This module, or a rebound copy when its model would apply different inputs.
        """
        del statistics, periods
        return self

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
    def training_step(self, model_state: Any, batches: NamedBatches) -> Mapping[str, jax.Array] | TrainingOutput:
        """Evaluate named unweighted losses for one compiled training step.

        Args:
            model_state: Explicit differentiable model parameter PyTree.
            batches: Fixed-structure named objective batches.

        Returns:
            Scalar losses keyed by :attr:`loss_names`, optionally accompanied by compiled diagnostics.
        """

    def log(
        self,
        name: str,
        value: Any,
        *,
        logger: bool = True,
        prog_bar: bool = False,
    ) -> None:
        """Route one host-side batch metric to loggers or progress displays.

        This method is available only while :class:`phijax.Trainer` executes :meth:`on_train_batch_end`. It records
        device values without transferring them to the host; the Trainer performs conversion at its configured
        logging or display cadence.

        Args:
            name: Non-empty complete metric name.
            value: Scalar host or device value. Arrays are accepted only when both destinations are disabled.
            logger: Whether configured experiment loggers receive the metric.
            prog_bar: Whether progress callbacks display the metric.

        Raises:
            RuntimeError: If called outside :meth:`on_train_batch_end`.
            TypeError: If `name`, `logger`, or `prog_bar` has an invalid type.
            ValueError: If `name` is empty or a routed value is not scalar.
        """
        collector = _ACTIVE_MODULE_METRICS.get()
        if collector is None:
            raise RuntimeError("`self.log()` is available only during `on_train_batch_end`.")
        if not isinstance(name, str):
            raise TypeError("Metric `name` must be a string.")
        if not name.strip():
            raise ValueError("Metric `name` must be non-empty.")
        if not isinstance(logger, bool):
            raise TypeError("Metric `logger` must be a boolean.")
        if not isinstance(prog_bar, bool):
            raise TypeError("Metric `prog_bar` must be a boolean.")
        if (logger or prog_bar) and not _metric_is_scalar(value):
            shape = getattr(value, "shape", None)
            raise ValueError(f"Logged metric `{name}` must be scalar, received shape {shape}.")
        collector.log(name, value, logger=logger, prog_bar=prog_bar)

    def format_training_metrics(
        self,
        total_loss: jax.Array,
        losses: Mapping[str, jax.Array],
        diagnostics: Mapping[str, jax.Array],
    ) -> dict[str, jax.Array]:
        """Format compiled training values for progress bars and experiment loggers.

        The combined loss becomes `train/loss`, each unweighted objective value becomes `train/loss/<name>`, and
        diagnostics become `train/<name>`. The host Trainer logs every scalar by default and displays only
        `train/loss` unless :meth:`log` overrides those destinations.

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


class PhiModule(BasePhiModule):
    """Describe a model factory and objective before Trainer-owned initialization.

    The user-facing instance is an immutable-in-practice blueprint. :class:`phijax.Trainer` calls :meth:`prepare_model`
    after the DataModule has exposed normalization statistics, producing a shallow bound copy with a pure model
    application and explicit model state. The original blueprint is never mutated.
    """

    def __init__(
        self,
        model: ModelFactory | InitializedModel,
        objective: Objective,
        *,
        name: str = "PINN",
    ) -> None:
        """Initialize an unbound module blueprint.

        Args:
            model: Lazy model factory or an existing :class:`phijax.InitializedModel`.
            objective: Objective producing named unweighted scalar losses.
            name: Non-empty human-readable module name.

        Raises:
            TypeError: If `model` cannot initialize an explicit-state model.
            ValueError: If `name` is empty or the objective exposes invalid loss names.
        """
        super().__init__(name=name)
        from phijax.models.contracts import InitializedModel

        if not isinstance(model, InitializedModel) and not callable(model):
            raise TypeError("`model` must be a model factory or `InitializedModel`.")
        resolved_names = tuple(objective.loss_names)
        if (
            not resolved_names
            or any(not isinstance(loss_name, str) or not loss_name.strip() for loss_name in resolved_names)
            or len(set(resolved_names)) != len(resolved_names)
        ):
            raise ValueError("The objective must expose unique non-empty loss names.")
        resolved_batch_keys = tuple(getattr(objective, "batch_keys", ()))
        if any(not isinstance(key, str) or not key.strip() for key in resolved_batch_keys) or len(
            set(resolved_batch_keys)
        ) != len(resolved_batch_keys):
            raise ValueError("The objective must expose unique non-empty batch keys.")
        self.model = model
        self.objective = objective
        self._loss_names = resolved_names
        self._batch_keys = resolved_batch_keys
        self._model_apply: ModelApply | None = None
        self._model_summary: ModelSummaryFunction | None = None
        # Build inputs recorded by `prepare_model` so `bind_inputs` can rebuild the apply closure.
        self._model_key: jax.Array | None = None
        self._precision: PrecisionPolicy | None = None
        self._input_statistics: InputStatistics | None = None
        self._input_periods: ResolvedInputPeriods | None = None

    @property
    def loss_names(self) -> tuple[str, ...]:
        """Return stable objective loss names.

        Returns:
            Ordered unique objective loss names.
        """
        return self._loss_names

    @property
    def batch_keys(self) -> tuple[str, ...]:
        """Return objective batch names in stable declaration order.

        Returns:
            Ordered unique batch names inferred from the objective.
        """
        return self._batch_keys

    def prepare_model(
        self,
        *,
        key: jax.Array,
        input_mean: jax.typing.ArrayLike | None,
        input_std: jax.typing.ArrayLike | None,
        precision: PrecisionPolicy,
        input_periods: InputPeriods | None = None,
    ) -> tuple[Self, Any]:
        """Initialize and bind the model without mutating this blueprint.

        Args:
            key: Model-parameter initialization key.
            input_mean: Optional per-coordinate normalization mean supplied by the DataModule.
            input_std: Optional per-coordinate normalization standard deviation supplied by the DataModule.
            precision: Trainer precision policy.
            input_periods: Optional physical period per input axis, from the DataModule or a restored checkpoint.

        Returns:
            Bound shallow module copy and its explicit initialized model state.

        Raises:
            TypeError: If the configured factory does not return :class:`phijax.InitializedModel`.
        """
        from phijax.models.contracts import InitializedModel

        initialized = self.model
        if not isinstance(initialized, InitializedModel):
            initialized = initialized(
                key=key,
                input_mean=input_mean,
                input_std=input_std,
                input_periods=input_periods,
                precision=precision,
            )
        if not isinstance(initialized, InitializedModel):
            raise TypeError("A model factory must return `InitializedModel`.")
        bound = copy(self)
        bound._model_apply = initialized.apply
        bound._model_summary = initialized.summary
        bound._model_key = key
        bound._precision = precision
        # A prebuilt `InitializedModel` ignores the statistics, so none are recorded.
        if isinstance(self.model, InitializedModel) or input_mean is None or input_std is None:
            bound._input_statistics = None
        else:
            bound._input_statistics = _host_statistics(input_mean, input_std)
        # The model reports the periods it applies, after its own configuration overrides.
        bound._input_periods = initialized.input_periods
        return bound, initialized.state

    @property
    def input_statistics(self) -> InputStatistics | None:
        """Return the input normalization statistics passed to the model factory.

        Returns:
            Host `float32` statistics, or `None` when the model was built without statistics or is a prebuilt
            :class:`phijax.InitializedModel`.
        """
        return self._input_statistics

    @property
    def input_periods(self) -> ResolvedInputPeriods | None:
        """Return the physical input periods the model applies.

        Returns:
            Sorted `(axis, period)` pairs reported by the model factory, or `None` without periodic features.
        """
        return self._input_periods

    def bind_inputs(self, statistics: InputStatistics | None, periods: InputPeriods | None) -> Self:
        """Return a module whose model applies `statistics` and `periods`.

        The model factory runs again with the original key and precision; its new parameter state is discarded. The
        factory reapplies its own period overrides, so the rebuilt model can match this module even when `periods`
        differ.

        Args:
            statistics: Normalization statistics to apply, or `None`.
            periods: Physical input periods to apply, or `None`.

        Returns:
            This module when its model already applies the same inputs or is a prebuilt
            :class:`phijax.InitializedModel`, otherwise a rebound copy. This module is never mutated.

        Raises:
            RuntimeError: If this blueprint has not been bound by :meth:`prepare_model`.
        """
        from phijax.models.contracts import InitializedModel

        if isinstance(self.model, InitializedModel):
            return self
        if self._model_apply is None or self._model_key is None or self._precision is None:
            raise RuntimeError("`bind_inputs()` requires a module bound by `prepare_model()`.")
        requested_statistics = None if statistics is None else _host_statistics(*statistics)
        requested_periods = as_input_periods(periods)
        if _same_statistics(self._input_statistics, requested_statistics) and requested_periods == self._input_periods:
            return self
        input_mean, input_std = (None, None) if requested_statistics is None else requested_statistics
        bound, _ = self.prepare_model(
            key=self._model_key,
            input_mean=input_mean,
            input_std=input_std,
            precision=self._precision,
            input_periods=None if requested_periods is None else dict(requested_periods),
        )
        if (
            _same_statistics(bound._input_statistics, self._input_statistics)
            and bound._input_periods == self._input_periods
        ):
            return self
        return bound

    def forward(self, model_state: Any, inputs: jax.Array) -> jax.Array:
        """Evaluate one model input with the configured application callable.

        Args:
            model_state: Explicit model parameter PyTree.
            inputs: One model input point.

        Returns:
            Model output for `inputs`.
        """
        return self._require_model_apply()(model_state, inputs)

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

    def training_step(self, model_state: Any, batches: NamedBatches) -> Mapping[str, jax.Array] | TrainingOutput:
        """Evaluate the configured objective's named unweighted losses.

        Args:
            model_state: Explicit differentiable model parameter PyTree.
            batches: Fixed-structure named objective batches.

        Returns:
            Scalar objective losses keyed by :attr:`loss_names`.
        """
        return self.objective.losses(self, model_state, batches)

    def on_train_batch_end(
        self,
        model_state: Any,
        context: PhiModuleContext,
    ) -> tuple[Any, Mapping[str, Any]]:
        """Declare the default logger and progress destinations for completed metrics.

        Every scalar is sent to experiment loggers. The total `train/loss`, individual `train/loss/<name>` losses,
        and `train/weight/<name>` balancer weights are displayed in that order by default. Array diagnostics remain
        available to callbacks without an implicit reduction or host transfer.

        Args:
            model_state: Updated explicit model state.
            context: Post-update context containing compiled metrics and diagnostics.

        Returns:
            Unchanged model state and complete metric mapping.
        """
        declared_names = {"train/loss"}
        self.log("train/loss", context.metrics["train/loss"], prog_bar=True)
        for loss_name in self.loss_names:
            loss_metric = f"train/loss/{loss_name}"
            if loss_metric in context.metrics:
                self.log(loss_metric, context.metrics[loss_metric], prog_bar=True)
                declared_names.add(loss_metric)
        for loss_name in self.loss_names:
            weight_metric = f"train/weight/{loss_name}"
            if weight_metric in context.metrics:
                self.log(weight_metric, context.metrics[weight_metric], prog_bar=True)
                declared_names.add(weight_metric)
        for name, value in context.metrics.items():
            if name not in declared_names and _metric_is_scalar(value):
                self.log(name, value)
        return model_state, context.metrics

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

    def _require_model_apply(self) -> ModelApply:
        """Return the bound pure model application.

        Returns:
            Pure model application created by :meth:`prepare_model`.

        Raises:
            RuntimeError: If this blueprint has not been bound by a Trainer.
        """
        if self._model_apply is None:
            raise RuntimeError("This `PhiModule` is uninitialized. Pass it to `Trainer.fit()` before evaluation.")
        return self._model_apply


def _same_statistics(first: InputStatistics | None, second: InputStatistics | None) -> bool:
    """Return whether two host statistics are both absent or exactly equal.

    Args:
        first: First statistics, or `None`.
        second: Second statistics, or `None`.

    Returns:
        Whether the statistics are interchangeable.
    """
    if first is None or second is None:
        return first is second
    return all(np.array_equal(a, b) for a, b in zip(first, second, strict=True))


def _host_statistics(mean: jax.typing.ArrayLike, std: jax.typing.ArrayLike) -> InputStatistics:
    """Copy input statistics to host `float32` NumPy arrays.

    Args:
        mean: Per-coordinate mean.
        std: Per-coordinate standard deviation.

    Returns:
        Host statistics that do not keep device buffers alive.
    """
    return InputStatistics(
        mean=np.asarray(jax.device_get(mean), dtype=np.float32),
        std=np.asarray(jax.device_get(std), dtype=np.float32),
    )


__all__ = ["BasePhiModule", "PhiModule", "PhiModuleContext"]
