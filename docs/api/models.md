# Models

PhiJAX includes Flax NNX models, but equations and objectives only need a pure, explicit-state callable. Before using
regular `jax.jit`, split the model and capture its graph definition in a closure or partial callable.

## Custom NNX architectures

`initialize_nnx_model` adapts any initialized Flax NNX module to :class:`phijax.models.InitializedModel`. Users only
define the architecture and an application-level factory that supplies its explicit initialization key:

```python
import jax
import jax.numpy as jnp
from flax import nnx

from phijax.models import InitializedModel, initialize_nnx_model
from phijax.training import PrecisionPolicy


class CoordinateNetwork(nnx.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        compute_dtype: object = jnp.float32,
        parameter_dtype: object = jnp.float32,
        output_dtype: object = jnp.float32,
        rngs: nnx.Rngs,
    ) -> None:
        self.output_dtype = output_dtype
        self.output = nnx.Linear(
            input_dim,
            output_dim,
            dtype=compute_dtype,
            param_dtype=parameter_dtype,
            rngs=rngs,
        )

    def __call__(self, inputs: jax.Array) -> jax.Array:
        return self.output(inputs).astype(self.output_dtype)


def build_coordinate_network(
    key: jax.Array,
    input_dim: int,
    output_dim: int,
    *,
    precision: str | PrecisionPolicy | None = None,
    **model_kwargs: object,
) -> InitializedModel:
    policy = PrecisionPolicy.from_name(precision or "32-true")
    resolved_kwargs = policy.apply_model_dtype_defaults(model_kwargs)
    model = CoordinateNetwork(input_dim, output_dim, rngs=nnx.Rngs(params=key), **resolved_kwargs)
    return initialize_nnx_model(
        model,
        example_inputs=jnp.zeros((1, input_dim), dtype=policy.derivative_dtype),
    )
```

The returned application merges the static graph with explicit state on each call. It works with JAX transformations
and PhiJAX equations without a base architecture or registry. `apply_model_dtype_defaults()` is optional. Use it when
the custom constructor accepts `parameter_dtype`, `compute_dtype`, and `output_dtype`. Values in `model_kwargs`
override the selected policy.

::: phijax.models.initialize_nnx_model

## MLP transformation order

The built-in MLP applies optional operations in this order:

```text
input normalization
-> periodic coordinate features
-> random Fourier features
-> hidden dense layers, activations, and dropout
-> output layer and optional output activation
```

`one_mlp_per_output=True` creates an independent scalar network per output. `weight_norm` and
`weight_factorization` are mutually exclusive. Stochastic dropout requires `deterministic=False` and an explicit
`dropout_key`.

```yaml
_target_: phijax.models.build_mlp
_partial_: true
input_dim: 2
output_dim: 1
hidden: [128, 128, 128, 128]
activation: tanh
input_norm: true
```

::: phijax.models.MLP

::: phijax.models.build_mlp

## Modified MLP

`ModifiedMLP` follows the gated architecture introduced for mitigating gradient-flow pathologies in PINNs. It creates
two shallow coordinate encodings, `U` and `V`, then uses each activated hidden layer as an element-wise interpolation
gate between them:

```text
U = activation(dense_u(features))
V = activation(dense_v(features))
H = activation(dense(H))
H = H * U + (1 - H) * V
```

Where applicable, the architecture supports the same normalization, coordinate features, output names, precision,
weight normalization, and random weight factorization as the standard MLP. It follows
[Wang, Teng, and Perdikaris (2021)](https://doi.org/10.1137/20M1318043).

```yaml
_target_: phijax.models.build_modified_mlp
_partial_: true
input_dim: 2
output_dim: 1
hidden_dim: 256
num_layers: 4
activation: tanh
input_norm: true
```

::: phijax.models.ModifiedMLP

::: phijax.models.build_modified_mlp

## PirateNet

`PirateNet` implements the Physics-Informed Residual Adaptive Network from
[Wang et al. (2024)](https://www.jmlr.org/papers/v25/24-0313.html). Each block contains three dense layers, two
modified-MLP gates, and a trainable scalar `alpha`:

```text
nonlinear = gated_three_layer_block(features, U, V)
features = alpha * nonlinear + (1 - alpha) * features
```

The default `alpha=0` makes every block an identity at initialization. Training increases the effective depth by
learning each coefficient. Random Fourier features are enabled by default and must produce `hidden_dim` features.
Because PhiJAX pairs cosine and sine features, the default samples `embed_dim=hidden_dim/2` frequencies. Random weight
factorization can be enabled separately.

The factory uses standard Glorot initialization. It does not apply the paper's optional physics-informed least-squares
initialization because that method depends on training data.

```yaml
_target_: phijax.models.build_pirate_net
_partial_: true
input_dim: 2
output_dim: 1
hidden_dim: 256
num_blocks: 4
activation: tanh
nonlinearity: 0.0
input_norm: true
fourier_features_kwargs:
  scale: 1.0
weight_factorization: true
weight_factorization_kwargs:
  mean: 0.5
  std: 0.1
```

::: phijax.models.PirateBlock

::: phijax.models.PirateNet

::: phijax.models.build_pirate_net

## Layers

`PeriodicFeatures` replaces each selected scalar coordinate by cosine and sine features, increasing the width by one
per selected axis. Configured frequencies are angular frequencies, so the physical period is `2*pi/abs(frequency)`.

`RandomFourierFeatures` returns `2 * embed_dim` features using the paired mapping
`x -> [cos(xB), sin(xB)]`. The construction follows the random-feature foundation of
[Rahimi and Recht (2007)](https://proceedings.neurips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html)
and its use for coordinate networks by
[Tancik et al. (2020)](https://proceedings.neurips.cc/paper/2020/hash/55053683268957697aa39fba6f231c68-Abstract.html).
PhiJAX stores the initially random projection `B` as a trainable parameter; keeping `B` fixed would recover the
classical random-feature interpretation.

`FactorizedDense` uses learned scale and direction parameters following
[Wang et al. (2022)](https://arxiv.org/abs/2210.01274). PhiJAX stores dense weights in
`[input_dim, output_dim]` layout, so its factorization is `W = V diag(g)`: each entry of `g` scales one output column.
With the common `[output_dim, input_dim]` layout, the same factorization is written `W = diag(g) V`.

::: phijax.models.PeriodicFeatures

::: phijax.models.RandomFourierFeatures

::: phijax.models.FactorizedDense

## Summary

::: phijax.models.tabulate_nnx_model
