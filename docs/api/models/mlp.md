# MLP

`MLP` is the standard fully connected PhiJAX architecture. Its factory supports input normalization, coordinate
features, dropout, independent output networks, and alternative dense-layer parameterizations.

## Transformation order

The architecture applies optional operations in this order:

```text
input normalization
-> periodic coordinate features
-> random Fourier features
-> hidden dense layers, activations, and dropout
-> output layer and optional output activation
```

Built-in architectures use the public `ActivationName` and `InitializationName` aliases. They also accept custom
`Activation` and `Initializer` callables.

| Option           | Built-in names                                                     |
| ---------------- | ------------------------------------------------------------------ |
| `activation`     | `"relu"`, `"leakyrelu"`, `"gelu"`, `"tanh"`, `"sigmoid"`, `"silu"` |
| `initialization` | `"kaiming_normal"`, `"xavier_uniform"`, `"trunc_normal"` or `None` |

`one_mlp_per_output=True` creates an independent scalar network per output. `weight_norm` and
`weight_factorization` are mutually exclusive. Stochastic dropout requires `deterministic=False` and an explicit
`dropout_key`.

## Factory example

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

See [Layers](layers.md) for periodic features, random Fourier features, and factorized dense layers.

## API reference

::: phijax.models.MLP

::: phijax.models.build_mlp
