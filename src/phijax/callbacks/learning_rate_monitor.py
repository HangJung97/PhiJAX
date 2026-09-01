from collections.abc import Callable, Mapping
from typing import Any, Literal

import jax
import jax.numpy as jnp

from phijax.callbacks.base import Callback, TrainerContext


class LearningRateMonitor(Callback):
    """Publish optimizer hyperparameters through the standard trainer metric stream.

    Optax schedules consume the pre-update optimizer count. A completed trainer step `n` therefore reports the value
    evaluated at `n - 1`, matching the hyperparameters used for that update. Unlike stateful PyTorch optimizers, an
    Optax transformation does not expose mutable parameter groups, so optional momentum and weight decay values are
    supplied explicitly by configuration.

    Attributes:
        schedule: Configured Optax-compatible step-indexed schedule.
        optimizer_name: Display name used to identify the Optax optimizer in metric keys.
        log_momentum: Whether to include the optimizer momentum coefficient.
        log_weight_decay: Whether to include the optimizer weight decay.
        log_key_prefix: String prepended verbatim to every metric name.
        logging_interval: Metric evaluation cadence, or `None` to follow the trainer logger cadence.
        momentum: Configured momentum scalar or schedule.
        weight_decay: Configured weight-decay scalar or schedule.
    """

    def __init__(
        self,
        schedule: Callable[[int], Any] | None,
        *,
        optimizer_name: str,
        log_momentum: bool = False,
        log_weight_decay: bool = False,
        log_key_prefix: str | None = "optimizer/",
        logging_interval: Literal["step", "epoch"] | None = None,
        momentum: Any = 0.0,
        weight_decay: Any = 0.0,
    ) -> None:
        """Initialize optimizer hyperparameter monitoring.

        Args:
            schedule: Optax-compatible callable receiving a zero-based optimizer step. `None` supports bootstrap Hydra
                composition but is rejected when the callback is used for fitting.
            optimizer_name: Non-empty display name added to `lr-<optimizer_name>`. PhiJAX requires this because Optax
                transformations do not retain an inspectable optimizer class name.
            log_momentum: Whether to log `momentum` as `lr-<optimizer_name>-momentum`.
            log_weight_decay: Whether to log `weight_decay` as `lr-<optimizer_name>-weight_decay`.
            log_key_prefix: Optional string prepended verbatim to every metric name. The default groups metrics under
                `optimizer/`; pass `None` to disable grouping.
            logging_interval: `"step"` evaluates every optimizer step, `"epoch"` evaluates at fit end, and `None`
                follows `trainer.log_every_n_steps` while always evaluating the final step.
            momentum: Optimizer momentum scalar or step-indexed schedule.
            weight_decay: Optimizer weight-decay scalar or step-indexed schedule.

        Raises:
            TypeError: If the schedule, logging flags, prefix, or optimizer hyperparameters have invalid types.
            ValueError: If `optimizer_name` is empty or `logging_interval` is unsupported.
        """
        if schedule is not None and not callable(schedule):
            raise TypeError("Learning-rate `schedule` must be callable or `None`.")
        if not isinstance(optimizer_name, str) or not optimizer_name.strip():
            raise ValueError("`optimizer_name` must be a non-empty string.")
        if not isinstance(log_momentum, bool):
            raise TypeError("`log_momentum` must be a boolean.")
        if not isinstance(log_weight_decay, bool):
            raise TypeError("`log_weight_decay` must be a boolean.")
        if log_key_prefix is not None and not isinstance(log_key_prefix, str):
            raise TypeError("`log_key_prefix` must be a string or `None`.")
        if logging_interval not in {None, "step", "epoch"}:
            raise ValueError("`logging_interval` must be `None`, `step`, or `epoch`.")
        _validate_scalar_source(momentum, "momentum")
        _validate_scalar_source(weight_decay, "weight_decay")
        self.schedule = schedule
        self.optimizer_name = optimizer_name
        self.log_momentum = log_momentum
        self.log_weight_decay = log_weight_decay
        self.log_key_prefix = log_key_prefix or ""
        self.logging_interval = logging_interval
        self.momentum = momentum
        self.weight_decay = weight_decay
        self._last_step: int | None = None

    def setup(self) -> None:
        """Verify an executable fit supplied a learning-rate schedule.

        Raises:
            RuntimeError: If the callback was instantiated without a schedule.
        """
        if self.schedule is None:
            raise RuntimeError("LearningRateMonitor requires a configured learning-rate `schedule`.")
        self._last_step = None

    def on_fit_start(self, context: TrainerContext) -> None:
        """Require an experiment logger before the first optimizer update.

        Args:
            context: Initial Trainer context exposing logger availability.

        Raises:
            RuntimeError: If the Trainer has no configured experiment logger.
        """
        if not context.has_logger:
            raise RuntimeError("Cannot use `LearningRateMonitor` with a Trainer that has no logger.")

    def training_metrics(self, context: TrainerContext) -> Mapping[str, jax.Array]:
        """Evaluate the learning rate used by the completed optimizer update.

        Args:
            context: Post-module context whose step counts completed optimizer updates.

        Returns:
            Learning-rate metric and any enabled optimizer hyperparameter metrics.

        Raises:
            RuntimeError: If no schedule was configured.
            ValueError: If a configured schedule does not return a scalar.
        """
        if self.schedule is None:
            raise RuntimeError("LearningRateMonitor requires a configured learning-rate `schedule`.")
        should_evaluate = (
            self.logging_interval == "step"
            or (self.logging_interval == "epoch" and context.is_fit_end)
            or (self.logging_interval is None and (context.should_log or context.is_fit_end))
        )
        if not should_evaluate or self._last_step == context.step:
            return {}
        schedule_step = max(context.step - 1, 0)
        metric_name = f"{self.log_key_prefix}lr-{self.optimizer_name}"
        metrics = {metric_name: _evaluate_scalar_source(self.schedule, schedule_step, "learning-rate schedule")}
        if self.log_momentum:
            metrics[f"{metric_name}-momentum"] = _evaluate_scalar_source(self.momentum, schedule_step, "momentum")
        if self.log_weight_decay:
            metrics[f"{metric_name}-weight_decay"] = _evaluate_scalar_source(
                self.weight_decay,
                schedule_step,
                "weight decay",
            )
        self._last_step = context.step
        return metrics

    def state_dict(self) -> Mapping[str, Any]:
        """Return the most recently emitted optimizer step.

        Returns:
            JSON-compatible monitor state.
        """
        return {"last_step": self._last_step}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore learning-rate emission bookkeeping.

        Args:
            state: Mapping containing an integer or null `last_step`.

        Raises:
            ValueError: If the mapping shape or saved step is invalid.
        """
        if set(state) != {"last_step"}:
            raise ValueError("LearningRateMonitor state must contain only `last_step`.")
        last_step = state["last_step"]
        if last_step is not None and (isinstance(last_step, bool) or not isinstance(last_step, int) or last_step < 0):
            raise ValueError("LearningRateMonitor `last_step` must be a nonnegative integer or `None`.")
        self._last_step = last_step


def _validate_scalar_source(source: Any, name: str) -> None:
    """Validate a scalar or callable optimizer hyperparameter source.

    Args:
        source: Candidate scalar or step-indexed callable.
        name: User-facing hyperparameter name.

    Raises:
        TypeError: If a non-callable value cannot be represented as a scalar JAX array.
        ValueError: If a non-callable value is not scalar.
    """
    if callable(source):
        return
    try:
        value = jnp.asarray(source)
    except (TypeError, ValueError) as error:
        raise TypeError(f"`{name}` must be a scalar or step-indexed callable.") from error
    if value.size != 1:
        raise ValueError(f"`{name}` must be scalar, received shape {value.shape}.")


def _evaluate_scalar_source(source: Any, step: int, name: str) -> jax.Array:
    """Evaluate and validate one optimizer hyperparameter at a schedule step.

    Args:
        source: Scalar value or step-indexed callable.
        step: Zero-based Optax schedule count.
        name: User-facing source description.

    Returns:
        Scalar JAX array.

    Raises:
        ValueError: If the evaluated value is not scalar.
    """
    value = jnp.asarray(source(step) if callable(source) else source)
    if value.size != 1:
        raise ValueError(f"{name.capitalize()} must return a scalar, received shape {value.shape}.")
    return value.reshape(())


__all__ = ["LearningRateMonitor"]
