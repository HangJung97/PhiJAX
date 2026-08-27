from typing import Any

import jax
import optax
from flax import struct

from phijax.training.precision import PrecisionPolicy


@struct.dataclass
class TrainState:
    """Carry all mutable arrays through compiled training computations.

    Attributes:
        model_state: Explicit differentiable NNX model state.
        optimizer_state: Optax transformation state aligned with `model_state`.
        balancer_state: Arbitrary JAX-compatible loss-balancer state.
        rng_key: Explicit JAX PRNG key for deterministic state evolution.
        step: Scalar integer training iteration stored on device.
        loss_scale: Current loss scale used by FP16 mixed precision.
        finite_steps: Consecutive finite-gradient update count used to grow `loss_scale`.
    """

    model_state: Any
    optimizer_state: optax.OptState
    balancer_state: Any
    rng_key: jax.Array
    step: jax.Array
    loss_scale: jax.Array
    finite_steps: jax.Array


def initialize_train_state(
    model_state: Any,
    optimizer: optax.GradientTransformation,
    balancer_state: Any,
    rng_key: jax.Array,
    precision: str | PrecisionPolicy = "32-true",
) -> TrainState:
    """Initialize the complete functional training state.

    Args:
        model_state: Explicit differentiable model parameter PyTree.
        optimizer: Optax transformation used to initialize optimizer slots.
        balancer_state: Initial loss-balancer weights and diagnostics.
        rng_key: JAX PRNG key to store for deterministic future operations.
        precision: Precision policy controlling parameter dtype and initial loss scaling.

    Returns:
        Training state with initialized optimizer slots and device step `0`.
    """
    policy = PrecisionPolicy.from_name(precision)
    model_state = policy.cast_model_state(model_state)
    return TrainState(
        model_state=model_state,
        optimizer_state=optimizer.init(model_state),
        balancer_state=balancer_state,
        rng_key=rng_key,
        step=jax.numpy.asarray(0, dtype=jax.numpy.int32),
        loss_scale=jax.numpy.asarray(policy.initial_loss_scale, dtype=jax.numpy.float32),
        finite_steps=jax.numpy.asarray(0, dtype=jax.numpy.int32),
    )
