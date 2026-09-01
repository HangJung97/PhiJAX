# Models

PhiJAX includes Flax NNX architectures, but equations and objectives only require a pure explicit-state callable. A
`ModelFactory` delays initialization until the Trainer has prepared data and selected precision. It returns an
`InitializedModel` containing an apply callable, explicit state, and optional summary.

## Model contracts

::: phijax.models.ModelFactory

::: phijax.models.InitializedModel

## Choose an architecture

| Architecture                | Use when                                                            |
| --------------------------- | ------------------------------------------------------------------- |
| [MLP](mlp.md)               | A conventional configurable coordinate network is sufficient        |
| [Modified MLP](modified.md) | Gated coordinate encodings may improve gradient flow                |
| [PirateNet](pirate-net.md)  | Training should gradually activate residual network depth           |
| [Layers](layers.md)         | A custom architecture needs PhiJAX feature or dense-layer utilities |

Each built-in architecture has a factory that returns `InitializedModel`. Bind architecture options with
`functools.partial` and pass the resulting factory to `PhiModule`.

## Custom NNX architectures

`initialize_nnx_model()` adapts an initialized Flax NNX module to `InitializedModel`. The application factory supplies
the explicit key, normalization statistics, and precision policy expected by the Trainer. See
[Build a custom model](../../guides/models.md) for a complete factory example and a non-NNX checklist.

::: phijax.models.initialize_nnx_model

## Model summaries

`tabulate_nnx_model()` creates the summary function used by built-in factories. Pass the result through
`InitializedModel.summary` when adapting a custom NNX architecture.

::: phijax.models.tabulate_nnx_model
