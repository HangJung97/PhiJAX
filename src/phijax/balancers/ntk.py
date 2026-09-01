from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from phijax.balancers.base import BalancerState, BalancerUpdatePlan
from phijax.core import BasePhiModule
from phijax.types import NamedBatches


def exact_ntk_trace(stream_fn: Callable[[Any], jax.Array], model_state: Any) -> jax.Array:
    """Compute an exact diagonal-NTK contribution without materializing a Jacobian.

    The function differentiates one scalar stream entry at a time, flattens the parameter-gradient PyTree, and sums its
    squared norm. A multi-feature stream therefore contributes the sum of its diagonal empirical NTK entries.
    :class:`ExactNTKBalancer` maps this calculation over samples in bounded chunks before averaging the resulting
    pointwise diagonal values.

    Args:
        stream_fn: Pure function mapping explicit model state to one residual array.
        model_state: Differentiable model parameter PyTree.

    Returns:
        Scalar `float32` sum of diagonal empirical-NTK entries across the flattened stream.
    """

    def flattened_stream(current_state: Any) -> jax.Array:
        """Flatten one residual stream while retaining parameter derivatives.

        Args:
            current_state: Differentiable model parameter PyTree.

        Returns:
            One-dimensional residual vector.
        """
        return stream_fn(current_state).reshape(-1)

    output_size = flattened_stream(model_state).size

    def accumulate(index: int, trace: jax.Array) -> jax.Array:
        """Accumulate the squared parameter-gradient norm for one residual entry.

        Args:
            index: Dynamic flattened residual index.
            trace: Exact trace accumulated for preceding residual entries.

        Returns:
            Updated scalar trace.
        """
        gradients = jax.grad(lambda current_state: flattened_stream(current_state)[index])(model_state)
        flattened_gradients, _ = ravel_pytree(gradients)
        flattened_gradients = flattened_gradients.astype(jnp.float32)
        row_norm = jnp.vdot(flattened_gradients, flattened_gradients).real
        return trace + row_norm

    return jax.lax.fori_loop(0, output_size, accumulate, jnp.asarray(0.0, dtype=jnp.float32))


def _pointwise_ntk_trace(
    module: BasePhiModule,
    name: str,
    model_state: Any,
    sample_batches: NamedBatches,
) -> jax.Array:
    """Compute one sample's diagonal-NTK contribution for one named loss.

    Args:
        module: Module exposing the named residual stream.
        name: Loss name selecting the residual stream.
        model_state: Differentiable model parameter PyTree.
        sample_batches: Named batches with their common leading sample axis removed by :func:`jax.vmap`.

    Returns:
        Scalar sum of diagonal empirical-NTK entries across the loss's residual features.
    """
    singleton_batches = jax.tree_util.tree_map(lambda value: value[jnp.newaxis, ...], sample_batches)
    stream_fn = partial(module.residual_stream, name, batches=singleton_batches)
    return exact_ntk_trace(stream_fn, model_state)


def _validate_kernel_chunk_size(kernel_chunk_size: int | None) -> int | None:
    """Validate a static kernel-sample chunk size.

    Args:
        kernel_chunk_size: Positive number of samples differentiated in parallel, or `None` for a full
            :func:`jax.vmap`.

    Returns:
        Unchanged validated chunk size.

    Raises:
        ValueError: If the chunk size is Boolean, zero, or negative.
    """
    if kernel_chunk_size is not None and (
        isinstance(kernel_chunk_size, bool) or not isinstance(kernel_chunk_size, int) or kernel_chunk_size < 1
    ):
        raise ValueError("`kernel_chunk_size` must be a positive integer or `None`.")
    return kernel_chunk_size


class ExactNTKBalancer:
    """Balance named losses with exact diagonal-NTK ratios and optional smoothing.

    For loss `i`, the balancer computes the exact diagonal empirical NTK independently at every sampled point and
    stores its pointwise mean as `mu_i`. The raw rule is `lambda_i = mean_j(mu_j) / mu_i`. A moving average blends each
    raw update with its previous value. Using empirical NTK statistics to diagnose and balance PINN loss terms follows
    Wang et al. (2022); PhiJAX uses mean-diagonal normalization and evaluates named residual streams sequentially to
    control peak memory.

    Attributes:
        momentum: Moving-average coefficient applied to previous weights.

    References:
        Wang, S., Yu, X., and Perdikaris, P. (2022). When and Why PINNs Fail to Train: A Neural Tangent Kernel
            Perspective. Journal of Computational Physics, 449, 110768.
    """

    def __init__(
        self,
        loss_names: Sequence[str],
        *,
        update_every_n_steps: int,
        kernel_size: int,
        kernel_chunk_size: int | None = 1,
        update_start_step: int | None = None,
        eps: float = 1.0e-10,
        moving_average_coefficient: float = 0.9,
        initial_weights: Mapping[str, float] | None = None,
    ) -> None:
        """Initialize exact-NTK balancing configuration.

        Args:
            loss_names: Unique non-empty sequence defining loss and trace ordering.
            update_every_n_steps: Positive optimizer-step interval between NTK updates.
            kernel_size: Positive diagnostic sample count requested for every objective batch.
            kernel_chunk_size: Positive number of diagnostic samples differentiated in parallel, or `None` for a full
                :func:`jax.vmap`.
            update_start_step: Nonnegative absolute optimizer step of the first update. `None` starts after one update
                interval.
            eps: Positive mean-diagonal floor used to avoid division by zero.
            moving_average_coefficient: Previous-weight coefficient in `[0, 1)`.
            initial_weights: Optional initial weights keyed by loss name; unspecified names use `1.0`.

        Raises:
            TypeError: If update scheduling or kernel-size values have invalid types.
            ValueError: If names are invalid, sizes are not positive, `eps` is not positive, or smoothing is outside
                `[0, 1)`.
        """
        if not loss_names or len(set(loss_names)) != len(loss_names):
            raise ValueError("`loss_names` must be non-empty and unique.")
        if isinstance(update_every_n_steps, bool) or not isinstance(update_every_n_steps, int):
            raise TypeError("`update_every_n_steps` must be an integer.")
        if update_every_n_steps < 1:
            raise ValueError("`update_every_n_steps` must be positive.")
        if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
            raise TypeError("`kernel_size` must be an integer.")
        if kernel_size < 1:
            raise ValueError("`kernel_size` must be positive.")
        if kernel_chunk_size is not None and (
            isinstance(kernel_chunk_size, bool) or not isinstance(kernel_chunk_size, int)
        ):
            raise TypeError("`kernel_chunk_size` must be an integer or `None`.")
        if update_start_step is not None and (
            isinstance(update_start_step, bool) or not isinstance(update_start_step, int)
        ):
            raise TypeError("`update_start_step` must be an integer or `None`.")
        if update_start_step is not None and update_start_step < 0:
            raise ValueError("`update_start_step` must be nonnegative.")
        kernel_chunk_size = _validate_kernel_chunk_size(kernel_chunk_size)
        if eps <= 0.0:
            raise ValueError("`eps` must be positive.")
        if not 0.0 <= moving_average_coefficient < 1.0:
            raise ValueError("`moving_average_coefficient` must be in `[0, 1)`.")
        configured = initial_weights or {}
        self.loss_names = tuple(loss_names)
        self.update_every_n_steps = update_every_n_steps
        self.update_start_step = update_every_n_steps if update_start_step is None else update_start_step
        self.kernel_size = kernel_size
        self.kernel_chunk_size = kernel_chunk_size
        self.eps = float(eps)
        self.momentum = float(moving_average_coefficient)
        self.initial_weights = jnp.asarray([configured.get(name, 1.0) for name in self.loss_names], dtype=jnp.float32)

    def initialize(self) -> BalancerState:
        """Create an initial balancer state.

        Returns:
            State with configured initial weights and zero mean-diagonal placeholders.
        """
        return BalancerState(weights=self.initial_weights, traces=jnp.zeros_like(self.initial_weights))

    def combine(
        self,
        losses: Mapping[str, jax.Array],
        state: BalancerState,
    ) -> jax.Array:
        """Apply the most recently computed NTK weights.

        Args:
            losses: Scalar loss arrays keyed by every configured loss name.
            state: Current NTK weights and mean-diagonal diagnostics.

        Returns:
            Weighted scalar total.

        Raises:
            KeyError: If a configured loss name is absent from `losses`.
        """
        components = jnp.stack([losses[name] for name in self.loss_names]).astype(jnp.float32)
        return jnp.vdot(state.weights, components)

    def diagnostics(self, state: BalancerState) -> Mapping[str, jax.Array]:
        """Expose current NTK weights and mean-diagonal estimates.

        Args:
            state: Current NTK balancer state.

        Returns:
            Scalar weights and traces keyed by loss name.
        """
        weights = {f"weight/{name}": state.weights[index] for index, name in enumerate(self.loss_names)}
        traces = {f"ntk/{name}": state.traces[index] for index, name in enumerate(self.loss_names)}
        return weights | traces

    def update_from_traces(self, traces: jax.Array, state: BalancerState) -> BalancerState:
        """Update smoothed weights from ordered mean diagonal-NTK values.

        Args:
            traces: Mean diagonal-NTK vector aligned with :attr:`loss_names`.
            state: Previous weights and mean-diagonal diagnostics.

        Returns:
            New stopped-gradient state containing nonnegative mean diagonals and smoothed weights. Zero entries
            preserve their previous weights; an all-zero vector leaves every weight unchanged.
        """
        nonnegative = jnp.maximum(traces.astype(jnp.float32), 0.0)
        mean_ntk = jnp.mean(nonnegative)
        safe_traces = jnp.maximum(nonnegative, self.eps)
        raw_weights = jnp.where(nonnegative > self.eps, mean_ntk / safe_traces, state.weights)
        updated = self.momentum * state.weights + (1.0 - self.momentum) * raw_weights
        weights = jnp.where(mean_ntk > self.eps, updated, state.weights)
        return BalancerState(weights=jax.lax.stop_gradient(weights), traces=jax.lax.stop_gradient(nonnegative))

    def update(
        self,
        module: BasePhiModule,
        model_state: Any,
        ntk_batches: NamedBatches,
        state: BalancerState,
        *,
        kernel_chunk_size: int | None = 1,
    ) -> BalancerState:
        """Compute pointwise diagonal NTKs one named stream at a time and update state.

        Args:
            module: Module exposing `residual_stream(name, model_state, batches)` independently of this balancer.
            model_state: Differentiable model parameter PyTree.
            ntk_batches: Fixed-shape named batches with a shared leading sample size, used only for NTK estimation.
            state: Previous balancer state.
            kernel_chunk_size: Positive number of kernel samples differentiated in parallel. `1` evaluates samples
                sequentially for minimum peak memory; `None` uses a full :func:`jax.vmap`.

        Returns:
            Updated weights and mean diagonal-NTK values in :attr:`loss_names` order.
        """
        kernel_chunk_size = _validate_kernel_chunk_size(kernel_chunk_size)
        traces = []
        for name in self.loss_names:
            pointwise_trace = partial(_pointwise_ntk_trace, module, name, model_state)
            if kernel_chunk_size is None:
                pointwise_diagonal = jax.vmap(pointwise_trace)(ntk_batches)
            else:
                pointwise_diagonal = jax.lax.map(
                    pointwise_trace,
                    ntk_batches,
                    batch_size=kernel_chunk_size,
                )
            traces.append(jnp.mean(pointwise_diagonal, dtype=jnp.float32))
        return self.update_from_traces(jnp.stack(traces), state)

    def make_update(
        self,
        module: BasePhiModule,
        *,
        kernel_chunk_size: int | None = 1,
    ) -> Callable[[Any, NamedBatches, BalancerState], BalancerState]:
        """Create the reusable compiled NTK update for fixed batch structures.

        Args:
            module: Module exposing one-at-a-time named residual streams.
            kernel_chunk_size: Positive number of kernel samples differentiated in parallel. `1` minimizes peak
                memory; `None` fully vectorizes across samples.

        Returns:
            JIT-compiled function accepting model state, NTK batches, and balancer state. New array values reuse the
            executable when shapes, dtypes, and PyTree structures stay fixed.

        Raises:
            ValueError: If `kernel_chunk_size` is not positive or `None`.
        """
        kernel_chunk_size = _validate_kernel_chunk_size(kernel_chunk_size)

        def update(
            model_state: Any,
            ntk_batches: NamedBatches,
            state: BalancerState,
        ) -> BalancerState:
            """Run one exact diagonal-NTK refresh.

            Args:
                model_state: Differentiable model parameter PyTree.
                ntk_batches: Fixed-shape named NTK batches sharing one leading sample size.
                state: Previous balancer weights and mean-diagonal diagnostics.

            Returns:
                Updated balancer state.
            """
            return self.update(
                module,
                model_state,
                ntk_batches,
                state,
                kernel_chunk_size=kernel_chunk_size,
            )

        return jax.jit(update)

    def build_update_plan(
        self,
        module: BasePhiModule,
        batch_keys: Sequence[str],
    ) -> BalancerUpdatePlan:
        """Describe an exact-NTK refresh with fixed diagnostic samples.

        Args:
            module: Module exposing one-at-a-time named residual streams.
            batch_keys: Stable objective batch keys sampled for every NTK refresh.

        Returns:
            Update plan containing scheduling, the compiled NTK update, and fixed diagnostic batch sizes.
        """
        return BalancerUpdatePlan(
            update=self.make_update(module, kernel_chunk_size=self.kernel_chunk_size),
            every_n_steps=self.update_every_n_steps,
            update_start_step=self.update_start_step,
            batch_sizes=dict.fromkeys(batch_keys, self.kernel_size),
        )
