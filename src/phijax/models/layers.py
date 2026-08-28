import math
from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from phijax.models.initialization import Initializer


class PeriodicFeatures(nnx.Module):
    """Replace selected coordinate axes with fixed cosine and sine features.

    Attributes:
        input_dim: Width of the physical input coordinate vector.
        axes: Unique coordinate axes replaced by periodic features.
        frequencies: Angular frequencies paired with :attr:`axes`.
        output_dim: Embedded width, equal to `input_dim + len(axes)`.
    """

    def __init__(
        self,
        input_dim: int,
        axes: Sequence[int],
        frequencies: Sequence[float],
    ) -> None:
        """Initialize deterministic periodic coordinate features.

        Each configured coordinate `x_i` is replaced by `(cos(w_i*x_i), sin(w_i*x_i))`, so its physical period is
        `2*pi/abs(w_i)`.

        Args:
            input_dim: Width of the physical input coordinate vector.
            axes: Unique coordinate axes to replace.
            frequencies: Nonzero finite angular frequencies paired with `axes`.

        Raises:
            ValueError: If dimensions, axes, or frequencies are invalid.
        """
        resolved_axes = tuple(int(axis) for axis in axes)
        resolved_frequencies = tuple(float(frequency) for frequency in frequencies)
        if input_dim < 1:
            raise ValueError("`input_dim` must be positive.")
        if not resolved_axes or len(resolved_axes) != len(resolved_frequencies):
            raise ValueError("`axes` and `frequencies` must contain the same positive number of entries.")
        invalid_axes = len(set(resolved_axes)) != len(resolved_axes) or any(
            axis < 0 or axis >= input_dim for axis in resolved_axes
        )
        if invalid_axes:
            raise ValueError("`axes` must contain unique valid input coordinate indices.")
        if any(not math.isfinite(frequency) or frequency == 0.0 for frequency in resolved_frequencies):
            raise ValueError("`frequencies` must contain finite nonzero angular frequencies.")
        self.input_dim = input_dim
        self.axes = resolved_axes
        self.frequencies = resolved_frequencies
        self.output_dim = input_dim + len(resolved_axes)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Embed configured axes while preserving leading input dimensions.

        Args:
            inputs: Coordinate array with final width :attr:`input_dim`.

        Returns:
            Embedded array with final width :attr:`output_dim`.

        Raises:
            ValueError: If the final input width does not match :attr:`input_dim`.
        """
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected final input width {self.input_dim}, got {inputs.shape[-1]}.")
        frequency_by_axis = dict(zip(self.axes, self.frequencies, strict=True))
        features: list[jax.Array] = []
        for axis in range(self.input_dim):
            coordinate = inputs[..., axis : axis + 1]
            if axis in frequency_by_axis:
                phase = frequency_by_axis[axis] * coordinate
                features.extend((jnp.cos(phase), jnp.sin(phase)))
            else:
                features.append(coordinate)
        return jnp.concatenate(features, axis=-1)


class RandomFourierFeatures(nnx.Module):
    """Map coordinates to paired cosine and sine Fourier features through a trainable random projection.

    The mapping `x -> [cos(xB), sin(xB)]` is related to random Fourier features for approximating shift-invariant
    kernels and to Fourier feature mappings for reducing the spectral bias of coordinate networks. PhiJAX samples `B`
    from a Gaussian distribution and stores it as a trainable :class:`nnx.Param`; classical random Fourier features
    instead keep the sampled projection fixed.

    Attributes:
        compute_dtype: Data type used for projection and trigonometric operations.
        kernel: Trainable Gaussian frequency matrix with shape `[input_dim, embed_dim]`.

    References:
        Rahimi, A. and Recht, B. (2007). Random Features for Large-Scale Kernel Machines. NeurIPS.
        Tancik, M. et al. (2020). Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional
            Domains. NeurIPS.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        *,
        scale: float = 1.0,
        compute_dtype: Any = jnp.float32,
        parameter_dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize a trainable Gaussian Fourier kernel.

        Args:
            input_dim: Number of input coordinate channels.
            embed_dim: Number of sampled frequencies. The output width is twice this value.
            scale: Standard deviation of the sampled frequencies.
            compute_dtype: Data type used for projection arithmetic.
            parameter_dtype: Data type used to store the trainable frequency matrix.
            rngs: NNX random streams used for initialization.

        Raises:
            ValueError: If `input_dim` or `embed_dim` is not positive.
        """
        if input_dim < 1 or embed_dim < 1:
            raise ValueError("`input_dim` and `embed_dim` must be positive.")
        self.compute_dtype = compute_dtype
        kernel = scale * jax.random.normal(rngs.params(), (input_dim, embed_dim), dtype=parameter_dtype)
        self.kernel = nnx.Param(kernel)

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Embed inputs through paired cosine and sine projections.

        Args:
            inputs: Coordinate array with final dimension `input_dim`.

        Returns:
            Embedded array with unchanged leading dimensions and final width `2 * embed_dim`.
        """
        projection = jnp.matmul(inputs.astype(self.compute_dtype), self.kernel[...].astype(self.compute_dtype))
        return jnp.concatenate((jnp.cos(projection), jnp.sin(projection)), axis=-1)


class FactorizedDense(nnx.Module):
    """Apply a random weight-factorized affine transform, `W = V diag(g)`.

    Each output channel has a learned positive scale `g_i` and direction vector `v_i`. With PhiJAX's dense weight
    layout `[input_dim, output_dim]`, scaling output columns gives `W = V diag(g)`. This is the transpose-layout
    equivalent of the commonly written `W = diag(g) V` for `[output_dim, input_dim]` weights.

    Attributes:
        compute_dtype: Data type used for affine computation.
        g: Trainable positive scale vector with shape `[output_dim]`.
        v: Trainable direction matrix with shape `[input_dim, output_dim]`.
        bias: Optional trainable additive bias with shape `[output_dim]`.

    References:
        Wang, S., Wang, H., Seidman, J. H., and Perdikaris, P. (2022). Random Weight Factorization Improves the Training
            of Continuous Neural Representations. arXiv:2210.01274.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        mean: float = 0.5,
        std: float = 0.1,
        use_bias: bool = True,
        kernel_init: Initializer | None = None,
        compute_dtype: Any = jnp.float32,
        parameter_dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize factorized parameters with a reconstructed Xavier-uniform weight.

        Args:
            input_dim: Input feature width.
            output_dim: Output feature width.
            mean: Mean used to sample the log scale.
            std: Standard deviation used to sample the log scale.
            use_bias: Whether to learn an additive bias.
            kernel_init: Initializer for the reconstructed base weight. Defaults to Xavier uniform.
            compute_dtype: Data type used for matrix multiplication.
            parameter_dtype: Data type used to store trainable parameters.
            rngs: NNX random streams used for initialization.

        Raises:
            ValueError: If `input_dim` or `output_dim` is not positive.
        """
        if input_dim < 1 or output_dim < 1:
            raise ValueError("Dense feature dimensions must be positive.")
        self.compute_dtype = compute_dtype
        parameter_dtype = jnp.dtype(parameter_dtype)
        scale = jnp.exp(mean + std * jax.random.normal(rngs.params(), (output_dim,), dtype=parameter_dtype))
        if kernel_init is None:
            limit = math.sqrt(6.0 / (input_dim + output_dim))
            base_weight = jax.random.uniform(
                rngs.params(),
                (input_dim, output_dim),
                dtype=parameter_dtype,
                minval=-limit,
                maxval=limit,
            )
        else:
            base_weight = kernel_init(rngs.params(), (input_dim, output_dim), parameter_dtype)
        self.g = nnx.Param(scale)
        self.v = nnx.Param(base_weight / scale[None, :])
        self.bias = nnx.Param(jnp.zeros((output_dim,), dtype=parameter_dtype)) if use_bias else None

    @property
    def weight(self) -> jax.Array:
        """Return the reconstructed dense weight.

        Returns:
            Weight array in `[input_dim, output_dim]` layout.
        """
        return self.v[...] * self.g[None, :]

    def __call__(self, inputs: jax.Array) -> jax.Array:
        """Apply the factorized affine transform.

        Args:
            inputs: Array whose final dimension is `input_dim`.

        Returns:
            Array with unchanged leading dimensions and final width `output_dim`.
        """
        values = inputs.astype(self.compute_dtype)
        weight = self.weight.astype(self.compute_dtype)
        output = jnp.matmul(values, weight)
        if self.bias is not None:
            output = output + self.bias[...].astype(self.compute_dtype)
        return output


def create_dense(
    input_dim: int,
    output_dim: int,
    *,
    kernel_init: Initializer | None,
    weight_norm: bool = False,
    weight_factorization: bool = False,
    weight_factorization_kwargs: Mapping[str, Any] | None = None,
    compute_dtype: Any,
    parameter_dtype: Any,
    rngs: nnx.Rngs,
) -> nnx.Module:
    """Construct a dense layer with shared PhiJAX parameterization policies.

    Args:
        input_dim: Input feature width.
        output_dim: Output feature width.
        kernel_init: Optional dense-kernel initializer.
        weight_norm: Whether to wrap a regular dense layer with NNX weight normalization.
        weight_factorization: Whether to use :class:`FactorizedDense`.
        weight_factorization_kwargs: Options forwarded to :class:`FactorizedDense`.
        compute_dtype: Data type used for affine arithmetic.
        parameter_dtype: Data type used for trainable parameters.
        rngs: NNX random streams used for parameter initialization.

    Returns:
        Configured dense NNX module.

    Raises:
        ValueError: If weight normalization and random weight factorization are both enabled.
    """
    if weight_norm and weight_factorization:
        raise ValueError("`weight_norm` and `weight_factorization` cannot both be enabled.")
    if weight_factorization:
        return FactorizedDense(
            input_dim,
            output_dim,
            kernel_init=kernel_init,
            compute_dtype=compute_dtype,
            parameter_dtype=parameter_dtype,
            rngs=rngs,
            **dict(weight_factorization_kwargs or {}),
        )
    linear_kwargs: dict[str, Any] = {
        "dtype": compute_dtype,
        "param_dtype": parameter_dtype,
        "rngs": rngs,
    }
    if kernel_init is not None:
        linear_kwargs["kernel_init"] = kernel_init
    linear = nnx.Linear(input_dim, output_dim, **linear_kwargs)
    return nnx.WeightNorm(linear, rngs=rngs) if weight_norm else linear
