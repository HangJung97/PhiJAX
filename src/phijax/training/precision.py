from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp

type PrecisionMode = Literal[
    "64-true",
    "32-true",
    "16-true",
    "bf16-true",
    "16-mixed",
    "bf16-mixed",
]


_ALIASES: dict[str, PrecisionMode] = {
    "float64": "64-true",
    "float32": "32-true",
    "float16": "16-true",
    "bfloat16": "bf16-true",
    "64": "64-true",
    "32": "32-true",
    "16": "16-mixed",
    "bf16": "bf16-mixed",
}


@dataclass(frozen=True, slots=True)
class PrecisionPolicy:
    """Describe parameter, arithmetic, output, and loss-scaling dtypes.

    Mixed modes retain `float32` parameters and public model outputs while using a lower precision for dense and
    embedding arithmetic. True modes use one dtype throughout. This distinction keeps coordinate derivatives and
    scalar PINN losses in `float32` for the recommended mixed-precision modes.

    Attributes:
        mode: Canonical Lightning-compatible precision name.
        parameter_dtype: Data type used to store trainable model parameters.
        compute_dtype: Data type used by supported model arithmetic.
        output_dtype: Data type returned by the model.
        derivative_dtype: Data type used for floating training batches and coordinate derivatives.
        dynamic_loss_scaling: Whether finite-gradient dynamic loss scaling is enabled.
        initial_loss_scale: Initial scale applied before differentiating an FP16 loss.
        growth_interval: Successful optimizer updates required before increasing the loss scale.
    """

    mode: PrecisionMode
    parameter_dtype: jnp.dtype
    compute_dtype: jnp.dtype
    output_dtype: jnp.dtype
    derivative_dtype: jnp.dtype
    dynamic_loss_scaling: bool
    initial_loss_scale: float
    growth_interval: int

    @classmethod
    def from_name(
        cls,
        precision: str | PrecisionPolicy = "32-true",
        *,
        derivative_dtype: Any | None = None,
        initial_loss_scale: float = 32768.0,
        growth_interval: int = 2000,
    ) -> PrecisionPolicy:
        """Resolve a Lightning-compatible precision mode.

        Args:
            precision: Canonical precision mode, supported alias, or existing policy.
            derivative_dtype: Optional override for floating input batches and coordinate derivatives.
            initial_loss_scale: Initial dynamic scale for `16-mixed` training.
            growth_interval: Successful updates between dynamic loss-scale growth events.

        Returns:
            Fully resolved immutable precision policy.

        Raises:
            ValueError: If a mode or loss-scaling option is invalid, or `64-true` is requested without JAX x64.
        """
        if isinstance(precision, PrecisionPolicy):
            return precision
        mode = cast(PrecisionMode, _ALIASES.get(precision, precision))
        supported = {"64-true", "32-true", "16-true", "bf16-true", "16-mixed", "bf16-mixed"}
        if mode not in supported:
            choices = ", ".join(sorted(supported))
            raise ValueError(f"Unknown precision mode `{precision}`. Available modes: {choices}.")
        if initial_loss_scale < 1.0:
            raise ValueError("`initial_loss_scale` must be at least `1.0`.")
        if growth_interval < 1:
            raise ValueError("`growth_interval` must be positive.")
        if mode == "64-true" and not jax.config.x64_enabled:
            raise ValueError("`64-true` requires `jax_enable_x64` before model or array initialization.")

        true_dtypes = {
            "64-true": jnp.dtype(jnp.float64),
            "32-true": jnp.dtype(jnp.float32),
            "16-true": jnp.dtype(jnp.float16),
            "bf16-true": jnp.dtype(jnp.bfloat16),
        }
        if mode in true_dtypes:
            parameter_dtype = compute_dtype = output_dtype = true_dtypes[mode]
        else:
            parameter_dtype = output_dtype = jnp.dtype(jnp.float32)
            compute_dtype = jnp.dtype(jnp.float16 if mode == "16-mixed" else jnp.bfloat16)
        resolved_derivative_dtype = jnp.dtype(derivative_dtype or output_dtype)
        return cls(
            mode=mode,
            parameter_dtype=parameter_dtype,
            compute_dtype=compute_dtype,
            output_dtype=output_dtype,
            derivative_dtype=resolved_derivative_dtype,
            dynamic_loss_scaling=mode == "16-mixed",
            initial_loss_scale=initial_loss_scale if mode == "16-mixed" else 1.0,
            growth_interval=growth_interval,
        )

    def cast_model_state(self, model_state: Any) -> Any:
        """Cast floating model-state leaves to the configured parameter dtype.

        Args:
            model_state: Parameter PyTree produced by a supported model.

        Returns:
            Model-state PyTree with floating leaves cast to `parameter_dtype`.
        """
        return jax.tree.map(lambda value: _cast_floating(value, self.parameter_dtype), model_state)

    def cast_batch(self, batch: Any) -> Any:
        """Cast floating batch leaves to the configured derivative dtype.

        Args:
            batch: Arbitrary batch PyTree.

        Returns:
            Batch PyTree preserving non-floating leaves and structure.
        """
        return jax.tree.map(lambda value: _cast_floating(value, self.derivative_dtype), batch)


def configure_precision(precision: str) -> None:
    """Apply process-wide JAX settings required before array initialization.

    Args:
        precision: Canonical precision mode or supported alias.

    Raises:
        ValueError: If `precision` is unknown.
    """
    mode = _ALIASES.get(precision, precision)
    if mode not in {"64-true", "32-true", "16-true", "bf16-true", "16-mixed", "bf16-mixed"}:
        raise ValueError(f"Unknown precision mode `{precision}`.")
    if mode == "64-true":
        jax.config.update("jax_enable_x64", True)


def _cast_floating(value: Any, dtype: jnp.dtype) -> Any:
    """Cast one floating array-like leaf while preserving other values.

    Args:
        value: Candidate PyTree leaf.
        dtype: Destination floating dtype.

    Returns:
        Cast value when it exposes a floating dtype, otherwise the original value.
    """
    value_dtype = getattr(value, "dtype", None)
    if value_dtype is not None and jnp.issubdtype(value_dtype, jnp.floating):
        return value.astype(dtype)
    return value


__all__ = ["PrecisionMode", "PrecisionPolicy", "configure_precision"]
