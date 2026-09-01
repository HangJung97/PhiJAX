# Models

PhiJAX includes Flax NNX models, but equations and objectives only need a pure, explicit-state callable. A
`ModelFactory` delays initialization until the Trainer has prepared data and selected precision. It returns an
`InitializedModel` with a pure apply callable, explicit state, and optional summary.

::: phijax.models.ModelFactory

::: phijax.models.InitializedModel

## Custom NNX architectures

`initialize_nnx_model` adapts an initialized Flax NNX module to `InitializedModel`. The application factory supplies
the explicit key, normalization statistics, and precision policy expected by the Trainer. See
[Build a custom model](../guides/models.md) for a complete factory example and a non-NNX checklist.

::: phijax.models.initialize_nnx_model

## MLP transformation order

Built-in architectures use the public `ActivationName` and `InitializationName` aliases. They also accept custom
`Activation` and `Initializer` callables.

| Option           | Built-in names                                                     |
| ---------------- | ------------------------------------------------------------------ |
| `activation`     | `"relu"`, `"leakyrelu"`, `"gelu"`, `"tanh"`, `"sigmoid"`, `"silu"` |
| `initialization` | `"kaiming_normal"`, `"xavier_uniform"`, `"trunc_normal"` or `None` |

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

```python
from functools import partial

from phijax.models import build_mlp

model_factory = partial(
    build_mlp,
    input_dim=2,
    output_dim=1,
    hidden=(128, 128, 128, 128),
    activation="tanh",
    input_norm=True,
)
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

```python
from functools import partial

from phijax.models import build_modified_mlp

model_factory = partial(
    build_modified_mlp,
    input_dim=2,
    output_dim=1,
    hidden_dim=256,
    num_layers=4,
    activation="tanh",
    input_norm=True,
)
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

```python
from functools import partial

from phijax.models import build_pirate_net

model_factory = partial(
    build_pirate_net,
    input_dim=2,
    output_dim=1,
    hidden_dim=256,
    num_blocks=4,
    activation="tanh",
    nonlinearity=0.0,
    input_norm=True,
    fourier_features_kwargs={"scale": 1.0},
    weight_factorization=True,
    weight_factorization_kwargs={"mean": 0.5, "std": 0.1},
)
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
