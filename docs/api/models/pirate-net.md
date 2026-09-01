# PirateNet

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

## Factory example

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

## API reference

::: phijax.models.PirateBlock

::: phijax.models.PirateNet

::: phijax.models.build_pirate_net
