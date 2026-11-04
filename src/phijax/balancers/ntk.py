from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Any, Literal, cast

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from phijax.balancers.base import BalancerState, BalancerUpdatePlan
from phijax.core import BasePhiModule
from phijax.types import NamedBatches

type KernelSize = int | Literal["all"]


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


def _device_blocks(batches: NamedBatches, devices: int) -> NamedBatches:
    """Rearrange rows split evenly over devices into `[rows_per_device, devices, ...]`.

    Device `d` holds rows `d * rows_per_device` to `(d + 1) * rows_per_device`, so the reshape keeps every row on its
    device and the swapped axis order lets a sequential map advance all devices together.

    Args:
        batches: Named batches whose leading axis holds `devices` contiguous blocks of rows.
        devices: Number of devices that split the leading axis.

    Returns:
        Batches with a leading row-block axis and a second device axis.
    """
    return jax.tree.map(
        lambda leaf: jnp.swapaxes(leaf.reshape(devices, leaf.shape[0] // devices, *leaf.shape[1:]), 0, 1),
        batches,
    )


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


def _validate_kernel_size(kernel_size: Any) -> KernelSize:
    """Validate one diagnostic sample count.

    Args:
        kernel_size: Candidate positive row count or `"all"`.

    Returns:
        Validated sample count.

    Raises:
        TypeError: If `kernel_size` is neither an integer nor `"all"`.
        ValueError: If an integer `kernel_size` is not positive.
    """
    if kernel_size == "all":
        return "all"
    if isinstance(kernel_size, bool) or not isinstance(kernel_size, int):
        raise TypeError("`kernel_size` must be a positive integer or 'all'.")
    if kernel_size < 1:
        raise ValueError("`kernel_size` must be positive.")
    return kernel_size


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
        kernel_size: KernelSize | Mapping[str, KernelSize],
        kernel_chunk_size: int | None = 1,
        resample: bool = False,
        update_start_step: int | None = None,
        eps: float = 1.0e-10,
        moving_average_coefficient: float = 0.9,
        initial_weights: Mapping[str, float] | None = None,
    ) -> None:
        """Initialize exact-NTK balancing configuration.

        Args:
            loss_names: Unique non-empty sequence defining loss and trace ordering.
            update_every_n_steps: Positive optimizer-step interval between NTK updates.
            kernel_size: Diagnostic sample count: a positive row count, or `"all"` for every row of a finite pool. A
                mapping sets the count per objective batch key and must name every batch key of the objective.
            kernel_chunk_size: Positive number of diagnostic samples differentiated in parallel, or `None` for a full
                :func:`jax.vmap`.
            resample: Whether to draw new diagnostic samples for every update instead of once per fit. Batches whose
                `kernel_size` is `"all"` contain the same rows either way.
            update_start_step: Nonnegative absolute optimizer step of the first update. `None` starts after one update
                interval.
            eps: Positive mean-diagonal floor used to avoid division by zero.
            moving_average_coefficient: Previous-weight coefficient in `[0, 1)`.
            initial_weights: Optional initial weights keyed by loss name; unspecified names use `1.0`.

        Raises:
            TypeError: If update scheduling, kernel-size, or `resample` values have invalid types.
            ValueError: If names are invalid, sizes are not positive, `eps` is not positive, or smoothing is outside
                `[0, 1)`.
        """
        if not loss_names or len(set(loss_names)) != len(loss_names):
            raise ValueError("`loss_names` must be non-empty and unique.")
        if isinstance(update_every_n_steps, bool) or not isinstance(update_every_n_steps, int):
            raise TypeError("`update_every_n_steps` must be an integer.")
        if update_every_n_steps < 1:
            raise ValueError("`update_every_n_steps` must be positive.")
        if isinstance(kernel_size, Mapping):
            if not kernel_size or any(not isinstance(name, str) or not name.strip() for name in kernel_size):
                raise ValueError("A `kernel_size` mapping must have non-empty string batch keys.")
            resolved_kernel_size: KernelSize | dict[str, KernelSize] = {
                name: _validate_kernel_size(size) for name, size in kernel_size.items()
            }
        else:
            resolved_kernel_size = _validate_kernel_size(kernel_size)
        if not isinstance(resample, bool):
            raise TypeError("`resample` must be a boolean.")
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
        self.kernel_size: KernelSize | dict[str, KernelSize] = resolved_kernel_size
        self.kernel_chunk_size = kernel_chunk_size
        self.resample = resample
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
            preserve their previous weights; an all-zero vector, or any NaN or infinite entry, leaves every weight
            unchanged.
        """
        nonnegative = jnp.maximum(traces.astype(jnp.float32), 0.0)
        mean_ntk = jnp.mean(nonnegative)
        safe_traces = jnp.maximum(nonnegative, self.eps)
        raw_weights = jnp.where(nonnegative > self.eps, mean_ntk / safe_traces, state.weights)
        updated = self.momentum * state.weights + (1.0 - self.momentum) * raw_weights
        # A NaN or infinite trace would corrupt every weight through the mean, so skip the whole update.
        weights = jnp.where((mean_ntk > self.eps) & jnp.all(jnp.isfinite(nonnegative)), updated, state.weights)
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
            module: Module exposing `residual_stream(name, model_state, batches)` and `stream_batch_keys(name)`
                independently of this balancer.
            model_state: Differentiable model parameter PyTree.
            ntk_batches: Fixed-shape named batches used only for NTK estimation. Each stream is evaluated over the
                batches named by `module.stream_batch_keys(name)`, which must share one leading row count.
            state: Previous balancer state.
            kernel_chunk_size: Positive number of kernel samples differentiated in parallel on each device. `1`
                evaluates samples sequentially for minimum peak memory; `None` uses a full :func:`jax.vmap`. With
                several devices, each device evaluates its own rows, so devices run in parallel either way.

        Returns:
            Updated weights and mean diagonal-NTK values in :attr:`loss_names` order.

        Raises:
            ValueError: If the batches read by one stream have different row counts.
        """
        kernel_chunk_size = _validate_kernel_chunk_size(kernel_chunk_size)
        traces = []
        for name in self.loss_names:
            # Map only over the batches this stream reads, whose rows are aligned; other keys may differ in size.
            stream_batch_keys = getattr(module, "stream_batch_keys", None)
            keys = tuple(cast(Sequence[str], stream_batch_keys(name))) if callable(stream_batch_keys) else ()
            keys = keys or tuple(ntk_batches)
            stream_batches = {key: ntk_batches[key] for key in keys}
            row_counts = {leaf.shape[0] for leaf in jax.tree.leaves(stream_batches)}
            if len(row_counts) != 1:
                raise ValueError(
                    f"NTK stream `{name}` reads batches {list(keys)} with different row counts {sorted(row_counts)}; "
                    "give them the same `kernel_size`."
                )
            pointwise_trace = partial(_pointwise_ntk_trace, module, name, model_state)
            (row_count,) = row_counts
            rows_per_device = self._rows_per_device(keys)
            devices = row_count // rows_per_device if rows_per_device and row_count % rows_per_device == 0 else 1
            if kernel_chunk_size is None:
                pointwise_diagonal = jax.vmap(pointwise_trace)(stream_batches)
            elif devices > 1:
                # A sequential loop over rows split across devices makes XLA gather every row onto every device.
                # Each device holds one contiguous block of `rows_per_device` rows, so loop over the block index and
                # vectorize over the device axis: every device differentiates only its own rows, in parallel.
                blocks = _device_blocks(stream_batches, devices)
                pointwise_diagonal = jax.lax.map(jax.vmap(pointwise_trace), blocks, batch_size=kernel_chunk_size)
            else:
                pointwise_diagonal = jax.lax.map(pointwise_trace, stream_batches, batch_size=kernel_chunk_size)
            traces.append(jnp.mean(pointwise_diagonal, dtype=jnp.float32))
        return self.update_from_traces(jnp.stack(traces), state)

    def _rows_per_device(self, keys: Sequence[str]) -> int | None:
        """Return the per-device diagnostic row count shared by the batches of one stream.

        Args:
            keys: Batch keys read by the stream.

        Returns:
            The common integer `kernel_size` of `keys`, or `None` when it is `all` or differs between them.
        """
        if isinstance(self.kernel_size, Mapping):
            sizes = {self.kernel_size.get(key, "all") for key in keys}
        else:
            sizes = {self.kernel_size}
        size = sizes.pop() if len(sizes) == 1 else "all"
        return size if isinstance(size, int) else None

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
        """Describe an exact-NTK refresh with diagnostic samples.

        Args:
            module: Module exposing one-at-a-time named residual streams.
            batch_keys: Stable objective batch keys sampled for every NTK refresh.

        Returns:
            Update plan containing scheduling, the compiled NTK update, and diagnostic batch sizes.

        Raises:
            ValueError: If a `kernel_size` mapping does not name exactly the objective batch keys.
        """
        if isinstance(self.kernel_size, Mapping):
            missing = sorted(set(batch_keys) - set(self.kernel_size))
            unknown = sorted(set(self.kernel_size) - set(batch_keys))
            if missing or unknown:
                raise ValueError(
                    f"`kernel_size` must name every objective batch key; missing {missing}, unknown {unknown}."
                )
            batch_sizes: dict[str, KernelSize] = {name: self.kernel_size[name] for name in batch_keys}
        else:
            batch_sizes = dict.fromkeys(batch_keys, self.kernel_size)
        return BalancerUpdatePlan(
            update=self.make_update(module, kernel_chunk_size=self.kernel_chunk_size),
            every_n_steps=self.update_every_n_steps,
            update_start_step=self.update_start_step,
            batch_sizes=batch_sizes,
            resample=self.resample,
        )
