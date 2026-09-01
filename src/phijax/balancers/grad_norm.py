from collections.abc import Callable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from phijax.balancers.base import BalancerState, BalancerUpdatePlan
from phijax.core import BasePhiModule
from phijax.metrics import TrainingOutput
from phijax.types import NamedBatches


def _loss_gradient_norm(
    module: BasePhiModule,
    name: str,
    model_state: Any,
    batches: NamedBatches,
) -> jax.Array:
    """Compute the Euclidean parameter-gradient norm of one scalar loss.

    Args:
        module: Module producing named unweighted scalar losses.
        name: Loss name selecting the differentiated scalar.
        model_state: Differentiable model parameter PyTree.
        batches: Fixed-structure batches used to evaluate the loss.

    Returns:
        Scalar `float32` Euclidean norm across every parameter-gradient leaf.
    """

    def scalar_loss(current_state: Any) -> jax.Array:
        """Select one unweighted scalar loss for differentiation.

        Args:
            current_state: Candidate differentiable model parameter PyTree.

        Returns:
            Scalar loss selected by `name`.
        """
        output = module.training_step(current_state, batches)
        losses = output.losses if isinstance(output, TrainingOutput) else output
        return losses[name]

    gradients = jax.grad(scalar_loss)(model_state)
    flattened_gradients, _ = ravel_pytree(gradients)
    return jnp.linalg.norm(flattened_gradients.astype(jnp.float32))


class GradNormBalancer:
    """Balance named losses with parameter-gradient norm ratios.

    The balancer computes the Euclidean parameter-gradient norm `g_i` of every unweighted scalar loss. Its raw weights
    are `lambda_i = mean_j(g_j) / (g_i + eps * mean_j(g_j))`, followed by an exponential moving average. Losses are
    differentiated one at a time so the update does not retain every loss Jacobian at once.

    This is a direct gradient-norm weighting policy rather than the training-rate-aware GradNorm algorithm for multitask
    networks.

    Attributes:
        momentum: Moving-average coefficient applied to previous weights.
    """

    def __init__(
        self,
        loss_names: Sequence[str],
        *,
        update_every_n_steps: int,
        update_start_step: int | None = None,
        eps: float = 1.0e-5,
        moving_average_coefficient: float = 0.9,
        initial_weights: Mapping[str, float] | None = None,
    ) -> None:
        """Initialize gradient-norm balancing configuration.

        Args:
            loss_names: Unique non-empty sequence defining loss and diagnostic ordering.
            update_every_n_steps: Positive optimizer-step interval between gradient-norm updates.
            update_start_step: Nonnegative absolute optimizer step of the first update. `None` starts after one update
                interval.
            eps: Nonnegative multiplier regularizing each gradient-norm denominator.
            moving_average_coefficient: Previous-weight coefficient in `[0, 1)`.
            initial_weights: Optional initial weights keyed by loss name; unspecified names use `1.0`.

        Raises:
            TypeError: If update scheduling values have invalid types.
            ValueError: If names are invalid, the update interval is not positive, `eps` is negative, or smoothing is
                outside `[0, 1)`.
        """
        if not loss_names or len(set(loss_names)) != len(loss_names):
            raise ValueError("`loss_names` must be non-empty and unique.")
        if isinstance(update_every_n_steps, bool) or not isinstance(update_every_n_steps, int):
            raise TypeError("`update_every_n_steps` must be an integer.")
        if update_every_n_steps < 1:
            raise ValueError("`update_every_n_steps` must be positive.")
        if update_start_step is not None and (
            isinstance(update_start_step, bool) or not isinstance(update_start_step, int)
        ):
            raise TypeError("`update_start_step` must be an integer or `None`.")
        if update_start_step is not None and update_start_step < 0:
            raise ValueError("`update_start_step` must be nonnegative.")
        if eps < 0.0:
            raise ValueError("`eps` must be nonnegative.")
        if not 0.0 <= moving_average_coefficient < 1.0:
            raise ValueError("`moving_average_coefficient` must be in `[0, 1)`.")
        configured = initial_weights or {}
        self.loss_names = tuple(loss_names)
        self.update_every_n_steps = update_every_n_steps
        self.update_start_step = update_every_n_steps if update_start_step is None else update_start_step
        self.eps = float(eps)
        self.momentum = float(moving_average_coefficient)
        self.initial_weights = jnp.asarray([configured.get(name, 1.0) for name in self.loss_names], dtype=jnp.float32)

    def initialize(self) -> BalancerState:
        """Create an initial balancer state.

        Returns:
            State with configured weights and zero gradient-norm diagnostics.
        """
        return BalancerState(weights=self.initial_weights, traces=jnp.zeros_like(self.initial_weights))

    def combine(
        self,
        losses: Mapping[str, jax.Array],
        state: BalancerState,
    ) -> jax.Array:
        """Apply the most recently computed gradient-norm weights.

        Args:
            losses: Scalar loss arrays keyed by every configured loss name.
            state: Current weights and gradient-norm diagnostics.

        Returns:
            Weighted scalar total.

        Raises:
            KeyError: If a configured loss name is absent from `losses`.
        """
        components = jnp.stack([losses[name] for name in self.loss_names]).astype(jnp.float32)
        return jnp.vdot(state.weights, components)

    def diagnostics(self, state: BalancerState) -> Mapping[str, jax.Array]:
        """Expose current loss weights and parameter-gradient norms.

        Args:
            state: Current gradient-norm balancer state.

        Returns:
            Scalar weights and gradient norms keyed by loss name.
        """
        weights = {f"weight/{name}": state.weights[index] for index, name in enumerate(self.loss_names)}
        norms = {f"grad_norm/{name}": state.traces[index] for index, name in enumerate(self.loss_names)}
        return weights | norms

    def update_from_grad_norms(self, grad_norms: jax.Array, state: BalancerState) -> BalancerState:
        """Update smoothed weights from ordered loss-gradient norms.

        Args:
            grad_norms: Gradient-norm vector aligned with :attr:`loss_names`.
            state: Previous weights and gradient-norm diagnostics.

        Returns:
            New stopped-gradient state. An all-zero norm vector leaves every weight unchanged.
        """
        nonnegative = jnp.maximum(grad_norms.astype(jnp.float32), 0.0)
        mean_grad_norm = jnp.mean(nonnegative)
        denominator = nonnegative + self.eps * mean_grad_norm
        raw_weights = mean_grad_norm / jnp.maximum(denominator, jnp.finfo(jnp.float32).tiny)
        updated = self.momentum * state.weights + (1.0 - self.momentum) * raw_weights
        weights = jnp.where(mean_grad_norm > 0.0, updated, state.weights)
        return BalancerState(weights=jax.lax.stop_gradient(weights), traces=jax.lax.stop_gradient(nonnegative))

    def update(
        self,
        module: BasePhiModule,
        model_state: Any,
        batches: NamedBatches,
        state: BalancerState,
    ) -> BalancerState:
        """Compute one loss-gradient norm at a time and update balancer state.

        Args:
            module: Module producing the named unweighted scalar losses.
            model_state: Differentiable model parameter PyTree.
            batches: Current fixed-structure training batches.
            state: Previous balancer state.

        Returns:
            Updated weights and gradient norms in :attr:`loss_names` order.
        """
        grad_norms = jnp.stack(
            [_loss_gradient_norm(module, name, model_state, batches) for name in self.loss_names],
        )
        return self.update_from_grad_norms(grad_norms, state)

    def make_update(
        self,
        module: BasePhiModule,
    ) -> Callable[[Any, NamedBatches, BalancerState], BalancerState]:
        """Create a reusable compiled gradient-norm update.

        Args:
            module: Module producing named unweighted scalar losses.

        Returns:
            JIT-compiled function accepting model state, current batches, and balancer state.
        """

        def update(
            model_state: Any,
            batches: NamedBatches,
            state: BalancerState,
        ) -> BalancerState:
            """Run one gradient-norm refresh.

            Args:
                model_state: Differentiable model parameter PyTree.
                batches: Current fixed-structure training batches.
                state: Previous balancer weights and diagnostics.

            Returns:
                Updated balancer state.
            """
            return self.update(module, model_state, batches, state)

        return jax.jit(update)

    def build_update_plan(
        self,
        module: BasePhiModule,
        batch_keys: Sequence[str],
    ) -> BalancerUpdatePlan:
        """Describe a gradient-norm refresh using the current training batches.

        Args:
            module: Module producing named unweighted scalar losses.
            batch_keys: Available objective batch keys, retained for the shared adaptive-balancer contract.

        Returns:
            Update plan that reuses the current training batches.
        """
        del batch_keys
        return BalancerUpdatePlan(
            update=self.make_update(module),
            every_n_steps=self.update_every_n_steps,
            update_start_step=self.update_start_step,
        )


__all__ = ["GradNormBalancer"]
