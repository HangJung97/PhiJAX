# API reference

The API reference lists PhiJAX's supported Python imports. Signatures, parameters, return values, and exceptions come
from source docstrings. Each page also explains ownership, array shapes, and lifecycle behavior. Optional configuration
integration is documented separately.

## Stability

Names re-exported by `phijax` or a package `__init__.py` are the public API. Private names beginning with `_` are
implementation details. Domain applications and their configuration trees live in separate project repositories.

PhiJAX is pre-1.0, so configuration names and Python interfaces may still change between minor releases. Migration
notes should accompany changes to public imports or required config paths.

The installed release is available as `phijax.__version__`.

## Core conventions

- Bound model, optimizer, balancer, PRNG, and step state are explicit JAX PyTrees.
- `PhiModule` blueprints retain model factories and objectives without mutable initialized parameters.
- A model application callable maps `(model_state, point)` to one output vector.
- Host data uses NumPy until the trainer-selected strategy performs device placement.
- Training arrays have fixed shapes and stable PyTree structures inside compiled computations.
- Objective, balancer, metric, callback, and checkpoint alignment uses ordered loss names.
- Every random operation accepts or derives an explicit JAX key.

## Common types

| Type                   | Contract                                                              |
| ---------------------- | --------------------------------------------------------------------- |
| `ArrayMapping`         | Field names mapped to JAX arrays                                      |
| `NamedBatches`         | Objective batch keys mapped to array mappings                         |
| `ModelApply`           | Explicit-state single-point model callable                            |
| `ResidualGroup`        | Residual arrays reduced into one scalar loss                          |
| `ResidualGroups`       | Residual groups aligned with static equation names                    |
| `ResidualStream`       | Literal `"residual"` or `"output"`                                    |
| `ModelSummaryFunction` | Callable rendering an explicit model state                            |
| `ResidualFunction`     | Configured equation callable consumed by an objective term            |
| `JaxDevice`            | Structural protocol for the stable device metadata used by strategies |

Package-specific public aliases include:

- `BatchSize`, `DevicePool`, `ArrayFormat`, and `DataStage` in `phijax.data`;
- `CallbackContext` in `phijax.callbacks`; and
- `PrecisionMode` in `phijax.training`.

::: phijax.types.JaxDevice

::: phijax.types.ModelSummaryFunction

::: phijax.types.ResidualFunction

## Import map

```python
from phijax import DataModule, ModelFactory, PhiModule, Trainer
from phijax.balancers import BalancerUpdatePlan, ExactNTKBalancer, GradNormBalancer, StaticLossBalancer
from phijax.callbacks import EarlyStopping, ModelCheckpoint, ProgressBar, RichModelSummary, RichProgressBar
from phijax.data import HostPool, NamedBatchSource, PhiDataModule, create_sampler
from phijax.evaluation import EvaluationResult, RegressionEvaluator
from phijax.equations import burgers_1d, polar_navier_stokes, residual_equation
from phijax.models import MLP, ModifiedMLP, PirateNet, build_mlp, build_modified_mlp, build_pirate_net
from phijax.models import initialize_nnx_model
from phijax.objectives import CompositeObjective, ResidualTerm
from phijax.training import OrbaxCheckpointIO, TrainState, TrainingPlan
from phijax.utils import RankedLogger, resolve_seed, seed_everything
```

The previous single-page reference remains available as an [API map](../api-reference.md).
