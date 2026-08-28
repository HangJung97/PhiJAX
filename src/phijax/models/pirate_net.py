import math
from collections.abc import Mapping, Sequence
from types import EllipsisType
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx
from jax.typing import ArrayLike

from phijax.models.activations import Activation, ActivationName, resolve_activation
from phijax.models.contracts import InitializedModel
from phijax.models.initialization import InitializationName, Initializer, resolve_initializer
from phijax.models.layers import PeriodicFeatures, RandomFourierFeatures, create_dense
from phijax.models.nnx_adapter import initialize_nnx_model
from phijax.training.precision import PrecisionPolicy


class PirateBlock(nnx.Module):
    """Apply one gated PirateNet block with an adaptive residual connection.

    Attributes:
        layers: Three hidden-width dense layers.
        alpha: Trainable scalar controlling the block's nonlinear contribution.
        activation: Point-wise activation applied after every dense layer.
        compute_dtype: Data type used for block arithmetic.
    """

    def __init__(
        self,
        hidden_dim: int,
        activation: Activation,
        *,
        nonlinearity: float = 0.0,
        kernel_init: Initializer | None = None,
        weight_factorization: bool = False,
        weight_factorization_kwargs: Mapping[str, Any] | None = None,
        compute_dtype: Any = jnp.float32,
        parameter_dtype: Any = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize an adaptive residual block.

        Args:
            hidden_dim: Shared width of block inputs, gates, and hidden activations.
            activation: Point-wise activation callable.
            nonlinearity: Initial value of the trainable residual coefficient `alpha`.
            kernel_init: Optional dense-kernel initializer.
            weight_factorization: Whether to use random weight-factorized dense layers.
            weight_factorization_kwargs: Options forwarded to :class:`FactorizedDense`.
            compute_dtype: Data type used for block arithmetic.
            parameter_dtype: Data type used to initialize trainable parameters.
            rngs: NNX random streams used for parameter initialization.

        Raises:
            ValueError: If `hidden_dim` is not positive or `nonlinearity` is not finite.
        """
        if hidden_dim < 1:
            raise ValueError("`hidden_dim` must be positive.")
        if not math.isfinite(nonlinearity):
            raise ValueError("`nonlinearity` must be finite.")
        self.activation = activation
        self.compute_dtype = jnp.dtype(compute_dtype)
        factorization_kwargs = dict(weight_factorization_kwargs or {})
        self.layers = nnx.List(
            [
                create_dense(
                    hidden_dim,
                    hidden_dim,
                    kernel_init=kernel_init,
                    weight_factorization=weight_factorization,
                    weight_factorization_kwargs=factorization_kwargs,
                    compute_dtype=self.compute_dtype,
                    parameter_dtype=parameter_dtype,
                    rngs=rngs,
                )
                for _ in range(3)
            ]
        )
        self.alpha = nnx.Param(jnp.asarray(nonlinearity, dtype=parameter_dtype))

    def __call__(self, inputs: jax.Array, gate_u: jax.Array, gate_v: jax.Array) -> jax.Array:
        """Evaluate the gated dense stack and adaptive identity interpolation.

        Args:
            inputs: Block inputs with final width `hidden_dim`.
            gate_u: First shallow latent gate with the same shape as `inputs`.
            gate_v: Second shallow latent gate with the same shape as `inputs`.

        Returns:
            Interpolation between the block inputs and nonlinear block output.
        """
        hidden = self.activation(self.layers[0](inputs))
        hidden = hidden * gate_u + (1.0 - hidden) * gate_v
        hidden = self.activation(self.layers[1](hidden))
        hidden = hidden * gate_u + (1.0 - hidden) * gate_v
        hidden = self.activation(self.layers[2](hidden))
        alpha = self.alpha[...].astype(self.compute_dtype)
        return alpha * hidden + (1.0 - alpha) * inputs


class PirateNet(nnx.Module):
    """Implement a Physics-Informed Residual Adaptive Network.

    PirateNet embeds coordinates into `hidden_dim` features, constructs two shallow latent gates, and applies a stack
    of three-layer adaptive residual blocks. Each block's scalar coefficient is normally initialized to zero, making
    every block an exact identity map at initialization before progressively learning a nonlinear contribution.

    Attributes:
        input_dim: Width of the physical input coordinate vector.
        output_dim: Width of the predicted output vector.
        output_names: Optional unique name for each scalar output.
        input_norm: Whether explicit input statistics are applied.
        periodic_embedding: Optional fixed periodic coordinate mapping.
        embedding: Optional random Fourier feature mapping.
        gate_u: Dense layer producing the first shallow latent gate.
        gate_v: Dense layer producing the second shallow latent gate.
        blocks: Adaptive residual blocks in evaluation order.
        output_layer: Final linear prediction layer.

    References:
        Wang, S., Li, B., Chen, Y., and Perdikaris, P. (2024). PirateNets: Physics-informed Deep Learning with Residual
            Adaptive Networks. Journal of Machine Learning Research, 25(402), 1-51.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_blocks: int = 2,
        activation: ActivationName | Activation = "tanh",
        activation_kwargs: Mapping[str, Any] | None = None,
        output_activation: ActivationName | Activation | None = None,
        output_activation_kwargs: Mapping[str, Any] | None = None,
        output_names: Sequence[str] | None = None,
        nonlinearity: float | Sequence[float] = 0.0,
        initialization: InitializationName | Initializer | None = "xavier_uniform",
        initialization_kwargs: Mapping[str, Any] | None = None,
        weight_factorization: bool = False,
        weight_factorization_kwargs: Mapping[str, Any] | None = None,
        periodic_features: bool = False,
        periodic_features_kwargs: Mapping[str, Any] | None = None,
        fourier_features: bool = True,
        fourier_features_kwargs: Mapping[str, Any] | None = None,
        input_norm: bool = False,
        input_norm_eps: float = 1.0e-8,
        compute_dtype: Any = jnp.float32,
        parameter_dtype: Any = jnp.float32,
        output_dtype: Any = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize a PirateNet architecture.

        Args:
            input_dim: Width of the physical input coordinate vector.
            output_dim: Width of the predicted output vector.
            hidden_dim: Shared width of coordinate embeddings, gates, and residual blocks.
            num_blocks: Number of adaptive residual blocks, each containing three dense layers.
            activation: Hidden activation name or callable.
            activation_kwargs: Keyword arguments bound to the hidden activation.
            output_activation: Optional output activation name or callable.
            output_activation_kwargs: Keyword arguments bound to the output activation.
            output_names: Optional unique names for scalar outputs.
            nonlinearity: Initial adaptive coefficient shared by all blocks or one value per block.
            initialization: Dense-kernel initializer name, callable, or `None` for the NNX default.
            initialization_kwargs: Keyword arguments forwarded to the initializer factory.
            weight_factorization: Whether to use random weight-factorized dense layers.
            weight_factorization_kwargs: Options for :class:`FactorizedDense`, including `mean`, `std`, and `use_bias`.
            periodic_features: Whether to replace selected coordinates with fixed cosine and sine features.
            periodic_features_kwargs: Options for :class:`PeriodicFeatures`, including `axes` and `frequencies`.
            fourier_features: Whether to use a trainable random Fourier coordinate embedding.
            fourier_features_kwargs: Options for :class:`RandomFourierFeatures`. `embed_dim` counts sampled frequencies,
                so the resulting feature width is `2 * embed_dim` and must equal `hidden_dim`.
            input_norm: Whether to apply explicit per-input mean and standard deviation arrays.
            input_norm_eps: Minimum standard deviation used during normalization.
            compute_dtype: Data type used for model arithmetic while parameters remain separately configurable.
            parameter_dtype: Data type used to initialize trainable parameters.
            output_dtype: Data type returned by the network.
            rngs: NNX random streams used for parameter initialization.

        Raises:
            ValueError: If dimensions, embeddings, output names, nonlinearities, or normalization options are invalid.
        """
        if input_dim < 1 or output_dim < 1 or hidden_dim < 1 or num_blocks < 1:
            raise ValueError("`input_dim`, `output_dim`, `hidden_dim`, and `num_blocks` must be positive.")
        if input_norm_eps < 0.0:
            raise ValueError("`input_norm_eps` must be non-negative.")
        names = tuple(output_names) if output_names is not None else None
        if names is not None and len(names) != output_dim:
            raise ValueError(f"`output_names` must contain {output_dim} names, got {len(names)}.")
        if names is not None and len(set(names)) != len(names):
            raise ValueError("`output_names` must contain unique names.")

        if isinstance(nonlinearity, int | float):
            nonlinearities = (float(nonlinearity),) * num_blocks
        else:
            nonlinearities = tuple(float(value) for value in nonlinearity)
        if len(nonlinearities) != num_blocks:
            raise ValueError(f"`nonlinearity` must contain {num_blocks} values, got {len(nonlinearities)}.")
        if any(not math.isfinite(value) for value in nonlinearities):
            raise ValueError("Every `nonlinearity` value must be finite.")

        resolved_activation = resolve_activation(activation, activation_kwargs)
        if resolved_activation is None:
            raise ValueError("PirateNet requires a hidden activation.")
        self.activation = resolved_activation
        self.output_activation = resolve_activation(output_activation, output_activation_kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_names = names
        self._output_index_map = {name: index for index, name in enumerate(names or ())}
        self.input_norm = input_norm
        self.input_norm_eps = input_norm_eps
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.output_dtype = jnp.dtype(output_dtype)
        parameter_dtype = jnp.dtype(parameter_dtype)

        if periodic_features:
            self.periodic_embedding = PeriodicFeatures(input_dim, **dict(periodic_features_kwargs or {}))
            embedding_width = self.periodic_embedding.output_dim
        else:
            self.periodic_embedding = None
            embedding_width = input_dim

        fourier_kwargs = dict(fourier_features_kwargs or {})
        if fourier_features:
            if "embed_dim" not in fourier_kwargs:
                if hidden_dim % 2:
                    raise ValueError("Default Fourier embedding requires an even `hidden_dim`.")
                fourier_kwargs["embed_dim"] = hidden_dim // 2
            self.embedding = RandomFourierFeatures(
                embedding_width,
                compute_dtype=self.compute_dtype,
                parameter_dtype=parameter_dtype,
                rngs=rngs,
                **fourier_kwargs,
            )
            embedding_width = 2 * int(fourier_kwargs["embed_dim"])
        else:
            self.embedding = None
        if embedding_width != hidden_dim:
            raise ValueError(
                f"PirateNet coordinate embeddings must have width `hidden_dim={hidden_dim}`, got {embedding_width}."
            )

        kernel_init = resolve_initializer(initialization, initialization_kwargs)
        factorization_kwargs = dict(weight_factorization_kwargs or {})
        dense_kwargs = {
            "kernel_init": kernel_init,
            "weight_factorization": weight_factorization,
            "weight_factorization_kwargs": factorization_kwargs,
            "compute_dtype": self.compute_dtype,
            "parameter_dtype": parameter_dtype,
            "rngs": rngs,
        }
        self.gate_u = create_dense(hidden_dim, hidden_dim, **dense_kwargs)
        self.gate_v = create_dense(hidden_dim, hidden_dim, **dense_kwargs)
        self.blocks = nnx.List(
            [
                PirateBlock(
                    hidden_dim,
                    self.activation,
                    nonlinearity=value,
                    kernel_init=kernel_init,
                    weight_factorization=weight_factorization,
                    weight_factorization_kwargs=factorization_kwargs,
                    compute_dtype=self.compute_dtype,
                    parameter_dtype=parameter_dtype,
                    rngs=rngs,
                )
                for value in nonlinearities
            ]
        )
        self.output_layer = create_dense(hidden_dim, output_dim, **dense_kwargs)

    def slice_output(self, name: str) -> tuple[EllipsisType, slice]:
        """Return an index tuple selecting one named scalar output.

        Args:
            name: Output name registered in `output_names`.

        Returns:
            An index preserving the selected scalar output dimension.

        Raises:
            ValueError: If output names were not configured.
            KeyError: If `name` is unknown.
        """
        if self.output_names is None:
            raise ValueError("`slice_output()` requires `output_names` to be configured.")
        if name not in self._output_index_map:
            raise KeyError(f"Unknown output name `{name}`. Available outputs: {self.output_names}.")
        index = self._output_index_map[name]
        return (..., slice(index, index + 1))

    def coordinate_features(
        self,
        inputs: jax.Array,
        input_mean: jax.Array | None = None,
        input_std: jax.Array | None = None,
    ) -> jax.Array:
        """Transform physical coordinates into fixed-width residual features.

        Args:
            inputs: Coordinate array with final width `input_dim`.
            input_mean: Optional per-input normalization mean.
            input_std: Optional per-input normalization standard deviation.

        Returns:
            Coordinate features with final width `hidden_dim`.

        Raises:
            ValueError: If input width or normalization arrays are invalid.
        """
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected final input width {self.input_dim}, got {inputs.shape[-1]}.")
        features = inputs.astype(self.compute_dtype)
        if self.input_norm:
            mean = jnp.zeros((self.input_dim,), dtype=self.compute_dtype) if input_mean is None else input_mean
            std = jnp.ones((self.input_dim,), dtype=self.compute_dtype) if input_std is None else input_std
            if mean.shape[-1] != self.input_dim or std.shape[-1] != self.input_dim:
                raise ValueError("Input statistics must match `input_dim`.")
            features = (features - mean) / jnp.maximum(std, self.input_norm_eps)
        if self.periodic_embedding is not None:
            features = self.periodic_embedding(features)
        if self.embedding is not None:
            features = self.embedding(features)
        return features

    def __call__(
        self,
        inputs: jax.Array,
        input_mean: jax.Array | None = None,
        input_std: jax.Array | None = None,
    ) -> jax.Array:
        """Evaluate PirateNet without mutating model state.

        Args:
            inputs: Input array with final width `input_dim`.
            input_mean: Optional per-input normalization mean.
            input_std: Optional per-input normalization standard deviation.

        Returns:
            Predictions with unchanged leading dimensions and final width `output_dim`.
        """
        features = self.coordinate_features(inputs, input_mean, input_std)
        gate_u = self.activation(self.gate_u(features))
        gate_v = self.activation(self.gate_v(features))
        hidden = features
        for block in self.blocks:
            hidden = block(hidden, gate_u, gate_v)
        outputs = self.output_layer(hidden)
        if self.output_activation is not None:
            outputs = self.output_activation(outputs)
        return outputs.astype(self.output_dtype)


def build_pirate_net(
    key: jax.Array,
    input_dim: int,
    output_dim: int,
    input_mean: ArrayLike,
    input_std: ArrayLike,
    *,
    precision: str | PrecisionPolicy | None = None,
    **model_kwargs: Any,
) -> InitializedModel:
    """Build normalized PirateNet through the generic initialized-model contract.

    Args:
        key: JAX PRNG key for deterministic parameter initialization.
        input_dim: Width of the physical input coordinate vector.
        output_dim: Width of the predicted output vector.
        input_mean: Array-like per-coordinate input normalization mean.
        input_std: Array-like per-coordinate input normalization standard deviation.
        precision: Optional model precision policy.
        **model_kwargs: Architecture options forwarded to :class:`PirateNet`.

    Returns:
        Pure model application, explicit NNX state, and model-summary callable.
    """
    policy = PrecisionPolicy.from_name(precision or "32-true")
    resolved_kwargs = policy.apply_model_dtype_defaults(model_kwargs)
    model = PirateNet(input_dim, output_dim, rngs=nnx.Rngs(params=key), **resolved_kwargs)
    return initialize_nnx_model(
        model,
        example_inputs=jnp.zeros((1, input_dim), dtype=policy.derivative_dtype),
        call_kwargs={"input_mean": jnp.asarray(input_mean), "input_std": jnp.asarray(input_std)},
    )


__all__ = ["PirateBlock", "PirateNet", "build_pirate_net"]
