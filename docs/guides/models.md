# Build a custom model

PhiJAX needs a [`ModelFactory`](../api/models.md#phijax.models.ModelFactory) that returns a pure model application, explicit array state,
and an optional summary function. The factory runs after the Trainer has selected precision and prepared input
statistics.

## Start with a built-in architecture

Use `functools.partial` to bind application dimensions and architecture settings. The Trainer supplies the remaining
factory arguments.

```python
from functools import partial

from phijax.models import build_mlp

model_factory = partial(
    build_mlp,
    input_dim=2,
    output_dim=1,
    hidden=(128, 128, 128),
    activation="tanh",
    input_norm=True,
)
```

Choose a built-in architecture by its behavior:

| Factory              | Use it when                                                            |
| -------------------- | ---------------------------------------------------------------------- |
| `build_mlp`          | A standard fully connected network is sufficient                       |
| `build_modified_mlp` | Gated coordinate encodings help a deeper PINN optimize                 |
| `build_pirate_net`   | Residual adaptive depth and Fourier features suit a harder PDE problem |

See the [model reference](../api/models.md) for every constructor option.

## Adapt a Flax NNX module

Define the NNX architecture normally, then adapt one initialized instance with
[`initialize_nnx_model()`](../api/models.md#phijax.models.initialize_nnx_model).

```python
import jax
import jax.numpy as jnp
from flax import nnx

from phijax.models import InitializedModel, initialize_nnx_model
from phijax.training import PrecisionName, PrecisionPolicy


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
    input_dim: int,
    output_dim: int,
    *,
    key: jax.Array,
    input_mean: jax.typing.ArrayLike | None,
    input_std: jax.typing.ArrayLike | None,
    precision: PrecisionName | PrecisionPolicy,
    **model_kwargs: object,
) -> InitializedModel:
    del input_mean, input_std
    policy = PrecisionPolicy.from_name(precision)
    options = policy.apply_model_dtype_defaults(model_kwargs)
    model = CoordinateNetwork(input_dim, output_dim, rngs=nnx.Rngs(params=key), **options)
    return initialize_nnx_model(
        model,
        example_inputs=jnp.zeros((1, input_dim), dtype=policy.derivative_dtype),
    )
```

The factory may ignore `input_mean` and `input_std` when the architecture does not normalize inputs.
`apply_model_dtype_defaults()` is optional. Use it when the constructor accepts parameter, compute, and output dtype
arguments. Explicit values in `model_kwargs` take precedence over the selected policy.

## Return a non-NNX model

A custom factory does not need Flax. Return [`InitializedModel`](../api/models.md#phijax.models.InitializedModel) with:

- `apply(model_state, inputs, ...)`, a pure callable accepted by JAX transformations;
- `state`, a JAX-compatible parameter PyTree; and
- `summary`, an optional callable that formats the explicit state.

Keep mutable arrays in the returned state. Do not capture trainable arrays in closures or mutate the module blueprint.
The optimizer and balancer remain Trainer-owned.

## Test the factory

Before fitting, test the factory with one fixed key and representative point batch:

1. Confirm the output shape and dtype.
2. Confirm repeated initialization with the same key produces equal state.
3. Differentiate the output with respect to both inputs and model state.
4. Run the apply function through `jax.jit` with the same PyTree structure used in training.
5. If stochastic behavior is supported, require an explicit runtime key.

Use identical parameters and inputs when comparing two architectures. Independent random initializations are not a
valid numerical parity test.

## Next steps

- [Models API](../api/models.md)
- [Precision](../api/precision.md)
- [Equations and objectives](objectives.md)
- [Train and predict](training.md)
