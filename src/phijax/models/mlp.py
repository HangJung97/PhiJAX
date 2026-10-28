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
from phijax.models.regularization import apply_dropout
from phijax.training.precision import PrecisionName, PrecisionPolicy


class MLP(nnx.Module):
    """Implement a dynamic multilayer perceptron with functional JAX state.

    Attributes:
        periodic_embedding: Optional fixed periodic coordinate mapping.
        embedding: Optional random Fourier feature mapping.
        networks: One shared network or one scalar-output network per output.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden: Sequence[int] | None = (128,),
        activation: ActivationName | Activation | None = "silu",
        activation_kwargs: Mapping[str, Any] | None = None,
        output_activation: ActivationName | Activation | None = None,
        output_activation_kwargs: Mapping[str, Any] | None = None,
        output_names: Sequence[str] | None = None,
        dropout: float = 0.0,
        initialization: InitializationName | Initializer | None = "xavier_uniform",
        initialization_kwargs: Mapping[str, Any] | None = None,
        one_mlp_per_output: bool = False,
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
        """Initialize a configurable MLP.

        Args:
            input_dim: Width of the input vector.
            output_dim: Width of the output vector.
            hidden: Width of each hidden layer. `None` or an empty sequence creates a direct input-to-output network.
            activation: Hidden activation name, callable, or `None`.
            activation_kwargs: Keyword arguments bound to each hidden activation.
            output_activation: Optional output activation name or callable.
            output_activation_kwargs: Keyword arguments bound to the output activation.
            output_names: Optional unique names for scalar outputs.
            dropout: Hidden-activation dropout rate. Stochastic calls require an explicit key.
            initialization: Dense-kernel initializer name, callable, or `None` for the NNX default.
            initialization_kwargs: Keyword arguments forwarded to the initializer factory.
            one_mlp_per_output: Whether to construct an independent scalar network for each output.
            weight_norm: Whether to wrap dense layers with NNX weight normalization.
            weight_factorization: Whether to use random weight-factorized dense layers.
            weight_factorization_kwargs: Options for :class:`FactorizedDense`, including `mean`, `std`, and `use_bias`.
            periodic_features: Whether to replace selected coordinates with fixed cosine and sine features.
            periodic_features_kwargs: Options for :class:`PeriodicFeatures`, including `axes` and `frequencies`.
            fourier_features: Whether to embed inputs with random Fourier features.
            fourier_features_kwargs: Options for :class:`RandomFourierFeatures`, including `embed_dim` and `scale`.
            input_norm: Whether to apply explicit per-input mean and standard deviation arrays.
            input_norm_eps: Minimum standard deviation used during normalization.
            compute_dtype: Data type used for model arithmetic while parameters remain `float32`.
            parameter_dtype: Data type used to initialize trainable parameters.
            output_dtype: Data type returned by the network.
            rngs: NNX random streams used for parameter initialization.

        Raises:
            TypeError: If `compute_dtype` cannot be resolved by JAX.
            ValueError: If dimensions, names, dropout, normalization, or layer options are invalid.
        """
        if input_dim < 1 or output_dim < 1:
            raise ValueError("`input_dim` and `output_dim` must be positive.")
        hidden_widths = tuple(hidden or ())
        if any(width < 1 for width in hidden_widths):
            raise ValueError("All `hidden` widths must be positive.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("`dropout` must satisfy `0 <= dropout < 1`.")
        if input_norm_eps < 0.0:
            raise ValueError("`input_norm_eps` must be non-negative.")
        if weight_norm and weight_factorization:
            raise ValueError("`weight_norm` and `weight_factorization` cannot both be enabled.")

        names = tuple(output_names) if output_names is not None else None
        if names is not None and len(names) != output_dim:
            raise ValueError(f"`output_names` must contain {output_dim} names, got {len(names)}.")
        if names is not None and len(set(names)) != len(names):
            raise ValueError("`output_names` must contain unique names.")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.output_names = names
        self._output_index_map = {name: index for index, name in enumerate(names or ())}
        self.input_norm = input_norm
        self.input_norm_eps = input_norm_eps
        self.dropout = dropout
        self.activation = resolve_activation(activation, activation_kwargs)
        self.output_activation = resolve_activation(output_activation, output_activation_kwargs)
        compute_dtype = jnp.dtype(compute_dtype)
        self.compute_dtype = compute_dtype
        self.output_dtype = jnp.dtype(output_dtype)
        parameter_dtype = jnp.dtype(parameter_dtype)

        if periodic_features:
            self.periodic_embedding = PeriodicFeatures(input_dim, **dict(periodic_features_kwargs or {}))
            embedding_width = self.periodic_embedding.output_dim
        else:
            self.periodic_embedding = None
            embedding_width = input_dim

        embedding_kwargs = dict(fourier_features_kwargs or {})
        if fourier_features:
            embed_dim = int(embedding_kwargs.pop("embed_dim", 32))
            self.embedding = RandomFourierFeatures(
                embedding_width,
                embed_dim,
                compute_dtype=compute_dtype,
                parameter_dtype=parameter_dtype,
                rngs=rngs,
                **embedding_kwargs,
            )
            embedding_width = 2 * embed_dim
        else:
            self.embedding = None

        kernel_init = resolve_initializer(initialization, initialization_kwargs)
        factorization_kwargs = dict(weight_factorization_kwargs or {})

        branch_output_widths = (1,) * output_dim if one_mlp_per_output else (output_dim,)
        self.networks = nnx.List(
            [
                nnx.List(
                    [
                        create_dense(
                            in_features,
                            out_features,
                            kernel_init=kernel_init,
                            weight_norm=weight_norm,
                            weight_factorization=weight_factorization,
                            weight_factorization_kwargs=factorization_kwargs,
                            compute_dtype=compute_dtype,
                            parameter_dtype=parameter_dtype,
                            rngs=rngs,
                        )
                        for in_features, out_features in zip(
                            (embedding_width, *hidden_widths),
                            (*hidden_widths, branch_output_width),
                            strict=True,
                        )
                    ]
                )
                for branch_output_width in branch_output_widths
            ]
        )

    def slice_output(self, name: str) -> tuple[EllipsisType, slice]:
        """Return an index tuple selecting one named scalar output.

        Args:
            name: Output name registered in `output_names`.

        Returns:
            An index of the form `(..., slice(i, i + 1))` preserving the scalar output dimension.

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

    def __call__(
        self,
        inputs: jax.Array,
        input_mean: jax.Array | None = None,
        input_std: jax.Array | None = None,
        *,
        deterministic: bool = True,
        dropout_key: jax.Array | None = None,
    ) -> jax.Array:
        """Evaluate the MLP without mutating model state.

        Args:
            inputs: Input array with final width `input_dim`.
            input_mean: Optional per-input normalization mean.
            input_std: Optional per-input normalization standard deviation.
            deterministic: Whether to disable configured dropout.
            dropout_key: Explicit PRNG key required for stochastic dropout.

        Returns:
            `float32` predictions with unchanged leading dimensions and final width `output_dim`.

        Raises:
            ValueError: If input width or normalization arrays are invalid, or stochastic dropout lacks a key.
        """
        if inputs.shape[-1] != self.input_dim:
            raise ValueError(f"Expected final input width {self.input_dim}, got {inputs.shape[-1]}.")
        values = inputs.astype(self.compute_dtype)
        if self.input_norm:
            mean = jnp.zeros((self.input_dim,), dtype=self.compute_dtype) if input_mean is None else input_mean
            std = jnp.ones((self.input_dim,), dtype=self.compute_dtype) if input_std is None else input_std
            if mean.shape[-1] != self.input_dim or std.shape[-1] != self.input_dim:
                raise ValueError("Input statistics must match `input_dim`.")
            values = (values - mean) / jnp.maximum(std, self.input_norm_eps)
        if self.periodic_embedding is not None:
            values = self.periodic_embedding(values)
        if self.embedding is not None:
            values = self.embedding(values)

        dropout_count = sum(len(network) - 1 for network in self.networks)
        if self.dropout and not deterministic:
            if dropout_key is None:
                raise ValueError("Stochastic dropout requires `dropout_key`.")
            dropout_keys = jax.random.split(dropout_key, dropout_count)
        else:
            dropout_keys = ()
        dropout_index = 0

        outputs: list[jax.Array] = []
        for network in self.networks:
            hidden = values
            for layer in network[:-1]:
                hidden = layer(hidden)
                if self.activation is not None:
                    hidden = self.activation(hidden)
                if self.dropout and not deterministic:
                    hidden = apply_dropout(hidden, self.dropout, dropout_keys[dropout_index])
                    dropout_index += 1
            output = network[-1](hidden)
            if self.output_activation is not None:
                output = self.output_activation(output)
            outputs.append(output)
        return jnp.concatenate(outputs, axis=-1).astype(self.output_dtype)


def build_mlp(
    key: jax.Array,
    input_dim: int,
    output_dim: int,
    input_mean: ArrayLike | None = None,
    input_std: ArrayLike | None = None,
    *,
    precision: PrecisionName | PrecisionPolicy | None = None,
    **model_kwargs: Any,
) -> InitializedModel:
    """Build a normalized MLP through the generic initialized-model contract.

    Args:
        key: JAX PRNG key for deterministic parameter initialization.
        input_dim: Width of the input vector.
        output_dim: Width of the predicted output vector.
        input_mean: Optional array-like per-coordinate input normalization mean.
        input_std: Optional array-like per-coordinate input normalization standard deviation.
        precision: Optional model precision policy.
        **model_kwargs: Architecture options forwarded to :class:`MLP`.

    Returns:
        Pure model application, explicit NNX state, and model-summary callable.
    """
    policy = PrecisionPolicy.from_name(precision or "32-true")
    resolved_kwargs = policy.apply_model_dtype_defaults(model_kwargs)
    model = MLP(input_dim, output_dim, rngs=nnx.Rngs(params=key), **resolved_kwargs)
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
