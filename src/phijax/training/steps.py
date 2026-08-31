from collections.abc import Callable
from dataclasses import replace
from typing import Any

import jax
import jax.numpy as jnp
import optax

from phijax.core import BasePhiModule
from phijax.metrics import TrainingOutput
from phijax.training.precision import PrecisionPolicy
from phijax.training.state import TrainState
from phijax.types import NamedBatches

type TrainStep = Callable[[TrainState, NamedBatches], tuple[TrainState, dict[str, jax.Array]]]


def make_train_step(
    module: BasePhiModule,
    balancer: Any,
    optimizer: optax.GradientTransformation,
    precision: str | PrecisionPolicy = "32-true",
) -> Callable[[TrainState, NamedBatches], tuple[TrainState, dict[str, jax.Array]]]:
    """Create one JIT-compiled loss, gradient, and Optax update.

    The returned function is created once and accepts new batch values without retracing as long as shapes, dtypes, and
    PyTree structures remain unchanged.

    Args:
        module: Module producing named unweighted scalar losses from explicit model state and batches.
        balancer: Balancer exposing stable `loss_names` and `combine(losses, state)`.
        optimizer: Initialized-compatible Optax gradient transformation.
        precision: Precision policy controlling batch dtype and FP16 dynamic loss scaling.

    Returns:
        JIT-compiled function mapping a :class:`TrainState` and fixed-structure named batches to updated state and
        named scalar metrics and fixed-shape diagnostics.
    """
    policy = PrecisionPolicy.from_name(precision)

    def train_step(
        state: TrainState,
        batches: NamedBatches,
    ) -> tuple[TrainState, dict[str, jax.Array]]:
        """Apply one functional optimization step.

        Args:
            state: Complete training state before the update.
            batches: Fixed-shape named objective batches.

        Returns:
            Updated training state plus named scalar metrics and fixed-shape diagnostics.
        """

        def loss_fn(model_state: Any) -> tuple[jax.Array, tuple[dict[str, jax.Array], dict[str, jax.Array]]]:
            """Evaluate the balanced objective for differentiation.

            Args:
                model_state: Candidate explicit model parameter state.

            Returns:
                Weighted scalar total plus named losses and diagnostics.
            """
            output = module.training_step(model_state, batches)
            if isinstance(output, TrainingOutput):
                losses = dict(output.losses)
                diagnostics = dict(output.diagnostics)
            else:
                losses = dict(output)
                diagnostics = {}
            total = balancer.combine(losses, state.balancer_state)
            return total, (losses, diagnostics)

        batches = policy.cast_batch(batches)

        def scaled_loss_fn(
            model_state: Any,
        ) -> tuple[jax.Array, tuple[jax.Array, dict[str, jax.Array], dict[str, jax.Array]]]:
            """Scale the differentiated loss while preserving unscaled metrics.

            Args:
                model_state: Candidate explicit model parameter state.

            Returns:
                Scaled loss and unscaled total plus named losses.
            """
            total, (losses, diagnostics) = loss_fn(model_state)
            return total * state.loss_scale, (total, losses, diagnostics)

        (_, (total, losses, module_diagnostics)), gradients = jax.value_and_grad(scaled_loss_fn, has_aux=True)(
            state.model_state
        )
        gradients = jax.tree.map(lambda gradient: gradient / state.loss_scale, gradients)
        gradients_finite = jnp.asarray(
            all(jnp.issubdtype(gradient.dtype, jnp.inexact) for gradient in jax.tree.leaves(gradients))
        ) & jnp.all(jnp.stack([jnp.all(jnp.isfinite(gradient)) for gradient in jax.tree.leaves(gradients)]))

        def apply_finite_update(_: None) -> tuple[Any, optax.OptState]:
            """Apply gradients when every leaf is finite.

            Args:
                _: Unused conditional operand.

            Returns:
                Updated model and optimizer state.
            """
            updates, optimizer_state = optimizer.update(gradients, state.optimizer_state, state.model_state)
            return optax.apply_updates(state.model_state, updates), optimizer_state

        def skip_nonfinite_update(_: None) -> tuple[Any, optax.OptState]:
            """Preserve state when FP16 scaling produces non-finite gradients.

            Args:
                _: Unused conditional operand.

            Returns:
                Unchanged model and optimizer state.
            """
            return state.model_state, state.optimizer_state

        should_update = gradients_finite | jnp.asarray(not policy.dynamic_loss_scaling)
        model_state, optimizer_state = jax.lax.cond(
            should_update,
            apply_finite_update,
            skip_nonfinite_update,
            operand=None,
        )
        successful_steps = jnp.where(gradients_finite, state.finite_steps + 1, 0)
        should_grow = successful_steps >= policy.growth_interval
        grown_scale = jnp.where(should_grow, state.loss_scale * 2.0, state.loss_scale)
        next_loss_scale = jnp.where(gradients_finite, grown_scale, jnp.maximum(state.loss_scale / 2.0, 1.0))
        next_loss_scale = jnp.where(policy.dynamic_loss_scaling, next_loss_scale, jnp.asarray(1.0, jnp.float32))
        successful_steps = jnp.where(should_grow, 0, successful_steps)
        rng_key, _ = jax.random.split(state.rng_key)
        next_state = replace(
            state,
            model_state=model_state,
            optimizer_state=optimizer_state,
            rng_key=rng_key,
            step=state.step + 1,
            loss_scale=next_loss_scale,
            finite_steps=successful_steps,
        )
        diagnostics = dict(module_diagnostics)
        balancer_diagnostics = dict(balancer.diagnostics(state.balancer_state))
        collisions = diagnostics.keys() & balancer_diagnostics.keys()
        if collisions:
            raise ValueError(f"Module and balancer diagnostics collide: {sorted(collisions)}.")
        diagnostics.update(balancer_diagnostics)
        diagnostics["precision/loss_scale"] = state.loss_scale
        diagnostics["precision/gradients_finite"] = gradients_finite.astype(jnp.float32)
        metrics = module.format_training_metrics(total, losses, diagnostics)
        return next_state, metrics

    return jax.jit(train_step)


__all__ = ["TrainStep", "make_train_step"]
