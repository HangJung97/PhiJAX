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
from phijax.training.precision import PrecisionName, PrecisionPolicy


class ModifiedMLP(nnx.Module):
    """Implement the gated modified MLP architecture used in modern PINN training.

    Two shallow latent encodings, `U` and `V`, are computed directly from the coordinate features. Every hidden layer
    then interpolates between these encodings using its activated output, improving gradient flow through deep PINNs.

    Attributes:
        input_dim: Width of the physical input coordinate vector.
        output_dim: Width of the predicted output vector.
        output_names: Optional unique name for each scalar output.
        input_norm: Whether explicit input statistics are applied.
        periodic_embedding: Optional fixed periodic coordinate mapping.
        embedding: Optional random Fourier feature mapping.
        gate_u: Dense layer producing the first shallow latent encoding.
        gate_v: Dense layer producing the second shallow latent encoding.
        layers: Gated hidden layers in evaluation order.
        output_layer: Final linear prediction layer.

    References:
        Wang, S., Teng, Y., and Perdikaris, P. (2021). Understanding and Mitigating Gradient Flow Pathologies in
            Physics-Informed Neural Networks. SIAM Journal on Scientific Computing, 43(5), A3055-A3081.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        activation: ActivationName | Activation = "tanh",
        activation_kwargs: Mapping[str, Any] | None = None,
        output_activation: ActivationName | Activation | None = None,
        output_activation_kwargs: Mapping[str, Any] | None = None,
        output_names: Sequence[str] | None = None,
        initialization: InitializationName | Initializer | None = "xavier_uniform",
        initialization_kwargs: Mapping[str, Any] | None = None,
        weight_norm: bool = False,
        weight_factorization: bool = False,
        weight_factorization_kwargs: Mapping[str, Any] | None = None,
        periodic_features: bool = False,
        periodic_features_kwargs: Mapping[str, Any] | None = None,
        fourier_features: bool = False,
        fourier_features_kwargs: Mapping[str, Any] | None = None,
        input_norm: bool = False,
        input_norm_eps: float = 1.0e-8,
        compute_dtype: Any = jnp.float32,
        parameter_dtype: Any = jnp.float32,
        output_dtype: Any = jnp.float32,
        *,
        rngs: nnx.Rngs,
    ) -> None:
        """Initialize a modified MLP.

        Args:
            input_dim: Width of the physical input coordinate vector.
            output_dim: Width of the predicted output vector.
            hidden_dim: Shared width of latent gates and hidden layers.
            num_layers: Number of gated hidden layers.
            activation: Hidden activation name or callable.
            activation_kwargs: Keyword arguments bound to the hidden activation.
            output_activation: Optional output activation name or callable.
            output_activation_kwargs: Keyword arguments bound to the output activation.
            output_names: Optional unique names for scalar outputs.
            initialization: Dense-kernel initializer name, callable, or `None` for the NNX default.
            initialization_kwargs: Keyword arguments forwarded to the initializer factory.
            weight_norm: Whether to apply NNX weight normalization to dense layers.
            weight_factorization: Whether to use random weight-factorized dense layers.
            weight_factorization_kwargs: Options for :class:`FactorizedDense`, including `mean`, `std`, and `use_bias`.
            periodic_features: Whether to replace selected coordinates with fixed cosine and sine features.
            periodic_features_kwargs: Options for :class:`PeriodicFeatures`, including `axes` and `frequencies`.
            fourier_features: Whether to embed coordinates with trainable random Fourier features.
            fourier_features_kwargs: Options for :class:`RandomFourierFeatures`, including `embed_dim` and `scale`.
            input_norm: Whether to apply explicit per-input mean and standard deviation arrays.
            input_norm_eps: Minimum standard deviation used during normalization.
            compute_dtype: Data type used for model arithmetic.
            parameter_dtype: Data type used to initialize trainable parameters.
            output_dtype: Data type returned by the network.
            rngs: NNX random streams used for parameter initialization.

        Raises:
            ValueError: If dimensions, names, normalization, activation, or dense-layer options are invalid.
        """
        if input_dim < 1 or output_dim < 1 or hidden_dim < 1 or num_layers < 1:
            raise ValueError("`input_dim`, `output_dim`, `hidden_dim`, and `num_layers` must be positive.")
        if input_norm_eps < 0.0:
            raise ValueError("`input_norm_eps` must be non-negative.")
        if weight_norm and weight_factorization:
            raise ValueError("`weight_norm` and `weight_factorization` cannot both be enabled.")
        names = tuple(output_names) if output_names is not None else None
        if names is not None and len(names) != output_dim:
            raise ValueError(f"`output_names` must contain {output_dim} names, got {len(names)}.")
        if names is not None and len(set(names)) != len(names):
            raise ValueError("`output_names` must contain unique names.")
        resolved_activation = resolve_activation(activation, activation_kwargs)
        if resolved_activation is None:
            raise ValueError("ModifiedMLP requires a hidden activation.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_names = names
        self._output_index_map = {name: index for index, name in enumerate(names or ())}
        self.input_norm = input_norm
        self.input_norm_eps = input_norm_eps
        self.activation = resolved_activation
        self.output_activation = resolve_activation(output_activation, output_activation_kwargs)
        self.compute_dtype = jnp.dtype(compute_dtype)
        self.output_dtype = jnp.dtype(output_dtype)
        parameter_dtype = jnp.dtype(parameter_dtype)

        if periodic_features:
            self.periodic_embedding = PeriodicFeatures(input_dim, **dict(periodic_features_kwargs or {}))
            embedding_width = self.periodic_embedding.output_dim
        else:
            self.periodic_embedding = None
            embedding_width = input_dim
        if fourier_features:
            fourier_kwargs = dict(fourier_features_kwargs or {})
            embed_dim = int(fourier_kwargs.pop("embed_dim", 32))
            self.embedding = RandomFourierFeatures(
                embedding_width,
                embed_dim,
                compute_dtype=self.compute_dtype,
                parameter_dtype=parameter_dtype,
                rngs=rngs,
                **fourier_kwargs,
            )
            embedding_width = 2 * embed_dim
        else:
            self.embedding = None

        kernel_init = resolve_initializer(initialization, initialization_kwargs)
        dense_kwargs = {
            "kernel_init": kernel_init,
            "weight_norm": weight_norm,
            "weight_factorization": weight_factorization,
            "weight_factorization_kwargs": dict(weight_factorization_kwargs or {}),
            "compute_dtype": self.compute_dtype,
            "parameter_dtype": parameter_dtype,
            "rngs": rngs,
        }
        self.gate_u = create_dense(embedding_width, hidden_dim, **dense_kwargs)
        self.gate_v = create_dense(embedding_width, hidden_dim, **dense_kwargs)
        layer_inputs = (embedding_width, *((hidden_dim,) * (num_layers - 1)))
        self.layers = nnx.List([create_dense(width, hidden_dim, **dense_kwargs) for width in layer_inputs])
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
        """Apply normalization and configured coordinate embeddings.

        Args:
            inputs: Coordinate array with final width `input_dim`.
            input_mean: Optional per-input normalization mean.
            input_std: Optional per-input normalization standard deviation.

        Returns:
            Embedded coordinate features consumed by the gates and hidden layers.

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
        """Evaluate the gated modified MLP without mutating model state.

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
        for layer in self.layers:
            hidden = self.activation(layer(hidden))
            hidden = hidden * gate_u + (1.0 - hidden) * gate_v
        outputs = self.output_layer(hidden)
        if self.output_activation is not None:
            outputs = self.output_activation(outputs)
        return outputs.astype(self.output_dtype)


def build_modified_mlp(
    key: jax.Array,
    input_dim: int,
    output_dim: int,
    input_mean: ArrayLike | None = None,
    input_std: ArrayLike | None = None,
    *,
    precision: PrecisionName | PrecisionPolicy | None = None,
    **model_kwargs: Any,
) -> InitializedModel:
    """Build a normalized modified MLP through the generic initialized-model contract.

    Args:
        key: JAX PRNG key for deterministic parameter initialization.
        input_dim: Width of the physical input coordinate vector.
        output_dim: Width of the predicted output vector.
        input_mean: Optional array-like per-coordinate input normalization mean.
        input_std: Optional array-like per-coordinate input normalization standard deviation.
        precision: Optional model precision policy.
        **model_kwargs: Architecture options forwarded to :class:`ModifiedMLP`.

    Returns:
        Pure model application, explicit NNX state, and model-summary callable.
    """
    policy = PrecisionPolicy.from_name(precision or "32-true")
    resolved_kwargs = policy.apply_model_dtype_defaults(model_kwargs)
    model = ModifiedMLP(input_dim, output_dim, rngs=nnx.Rngs(params=key), **resolved_kwargs)
    if (input_mean is None) != (input_std is None):
        raise ValueError("`input_mean` and `input_std` must either both be provided or both be `None`.")
    call_kwargs = None
    if input_mean is not None and input_std is not None:
        call_kwargs = {"input_mean": jnp.asarray(input_mean), "input_std": jnp.asarray(input_std)}
    return initialize_nnx_model(
        model,
        example_inputs=jnp.zeros((1, input_dim), dtype=policy.derivative_dtype),
        call_kwargs=call_kwargs,
    )


__all__ = ["ModifiedMLP", "build_modified_mlp"]
