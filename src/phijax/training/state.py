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
        rng_key: Explicit PRNG stream for stochastic model execution.
        sampling_key: Persistent PRNG stream for DataModule-owned training batches.
        balancer_key: Persistent PRNG stream for adaptive-balancer diagnostics.
        step: Scalar integer training iteration stored on device.
        loss_scale: Current loss scale used by FP16 mixed precision.
        finite_steps: Consecutive finite-gradient update count used to grow `loss_scale`.
    """

    model_state: Any
    optimizer_state: optax.OptState
    balancer_state: Any
    rng_key: jax.Array
    sampling_key: jax.Array
    balancer_key: jax.Array
    step: jax.Array
    loss_scale: jax.Array
    finite_steps: jax.Array


def initialize_train_state(
    model_state: Any,
    optimizer: optax.GradientTransformation,
    balancer_state: Any,
    key: jax.Array,
    precision: str | PrecisionPolicy = "32-true",
    *,
    sampling_key: jax.Array | None = None,
    balancer_key: jax.Array | None = None,
) -> TrainState:
    """Initialize the complete functional training state.

    Args:
        model_state: Explicit differentiable model parameter PyTree.
        optimizer: Optax transformation used to initialize optimizer slots.
        balancer_state: Initial loss-balancer weights and diagnostics.
        key: Root training key, or the model-runtime key when both explicit persistent keys are supplied.
        precision: Precision policy controlling parameter dtype and initial loss scaling.
        sampling_key: Optional explicit DataModule sampling key.
        balancer_key: Optional explicit adaptive-balancer diagnostic key.

    Returns:
        Training state with initialized optimizer slots and device step `0`.

    Raises:
        ValueError: If only one explicit persistent key is supplied.
    """
    policy = PrecisionPolicy.from_name(precision)
    model_state = policy.cast_model_state(model_state)
    rng_key = key
    if sampling_key is None:
        if balancer_key is not None:
            raise ValueError("`sampling_key` and `balancer_key` must either both be supplied or both be omitted.")
        rng_key, resolved_sampling_key, resolved_balancer_key = jax.random.split(key, 3)
    elif balancer_key is None:
        raise ValueError("`sampling_key` and `balancer_key` must either both be supplied or both be omitted.")
    else:
        resolved_sampling_key = sampling_key
        resolved_balancer_key = balancer_key
    return TrainState(
        model_state=model_state,
        optimizer_state=optimizer.init(model_state),
        balancer_state=balancer_state,
        rng_key=rng_key,
        sampling_key=resolved_sampling_key,
        balancer_key=resolved_balancer_key,
        step=jax.numpy.asarray(0, dtype=jax.numpy.int32),
        loss_scale=jax.numpy.asarray(policy.initial_loss_scale, dtype=jax.numpy.float32),
        finite_steps=jax.numpy.asarray(0, dtype=jax.numpy.int32),
    )
