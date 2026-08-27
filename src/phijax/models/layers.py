import math
from collections.abc import Sequence
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
    """Map coordinates to paired cosine and sine random Fourier features.

    Attributes:
        compute_dtype: Data type used for projection and trigonometric operations.
        kernel: Trainable Gaussian frequency matrix with shape `[input_dim, embed_dim]`.
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
    """Apply a random weight-factorized affine transform, `W = diag(g) V`.

    Attributes:
        compute_dtype: Data type used for affine computation.
        g: Trainable positive scale vector with shape `[output_dim]`.
        v: Trainable direction matrix with shape `[input_dim, output_dim]`.
        bias: Optional trainable additive bias with shape `[output_dim]`.
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
