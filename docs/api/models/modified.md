# Modified MLP

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
weight normalization, and random weight factorization as the standard [MLP](mlp.md). It follows
[Wang, Teng, and Perdikaris (2021)](https://doi.org/10.1137/20M1318043).

## Factory example

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

## API reference

::: phijax.models.ModifiedMLP

::: phijax.models.build_modified_mlp
