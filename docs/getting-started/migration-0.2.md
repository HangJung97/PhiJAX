# Migrating from 0.1 to 0.2

PhiJAX 0.2 makes `Trainer.fit()` the common application API. It initializes the model and functional state, infers
batch routing, compiles the update, and owns DataModule sources. The explicit 0.1 execution path remains available
under names that describe its state-level contract.

## API replacements

| PhiJAX 0.1                                                       | PhiJAX 0.2                                                          |
| ---------------------------------------------------------------- | ------------------------------------------------------------------- |
| `PhiModule(model_apply, objective, model_summary=...)`           | `PhiModule(model_factory, objective)`                               |
| `initialize_state()` + `compile_train_step()` + `TrainingPlan()` | `Trainer.fit(..., optimizer=..., seed=...)`                         |
| `Trainer.fit(module, plan, state, ...)`                          | `Trainer.fit_state(module, plan, state, ...)`                       |
| `Trainer.predict(module, state, ...)`                            | `Trainer.predict_state(module, state, ...)`                         |
| Explicit post-fit module and state                               | `Trainer.predict(fit_result, ...)`                                  |
| `resume_latest(...)`                                             | `fit(..., ckpt_path="last")`                                        |
| `MetricRoute` and `configure_metric_routes()`                    | `self.log(..., logger=..., prog_bar=...)` in `on_train_batch_end()` |
| `from phijax.module import PhiModule`                            | `from phijax import PhiModule`                                      |
| `from phijax.module import BasePhiModule`                        | `from phijax.core import BasePhiModule`                             |
| `from phijax.training.trainer import FitResult`                  | `from phijax.training import FitResult`                             |

Top-level imports are the recommended application API. PhiJAX 0.2 removes the old `phijax.module` path; custom module
implementations that need a deep import should use `phijax.core`.

## Lazy model initialization

Bind architecture options without creating parameters:

```python
from functools import partial

from phijax import PhiModule
from phijax.models import build_mlp

model_factory = partial(
    build_mlp,
    input_dim=2,
    output_dim=1,
    hidden=(64, 64),
    input_norm=True,
)
module = PhiModule(model_factory, objective)
result = trainer.fit(module, datamodule=data_module, optimizer=optimizer, seed=0)
```

The Trainer supplies `key`, optional `input_mean` and `input_std`, and `precision`. Override
`PhiDataModule.input_statistics()` to enable normalization; it returns `None` by default. Custom factories return an
`InitializedModel` and do not need to use Flax NNX.

## Objective and finite-data shorthands

When batch routing follows equation names, replace explicit terms with:

```python
objective = CompositeObjective.from_equations(
    {
        "initial": initial_equation,
        "boundary": boundary_equation,
        "pde": pde_equation,
    }
)
```

`ResidualTerm` now takes its equation first: `ResidualTerm(equation, batch_key="pde")`. Explicit `names` and
`ntk_stream` remain keyword-only overrides.

For aligned finite pools, use:

```python
return NamedBatchSource.from_pools(self.pools, self.batch_sizes, key, names=batch_keys)
```

Omitted `HostPool.targets`, `aux`, `metadata`, `reference_shape`, and `flat_index` now receive safe defaults.

## State and checkpoints

`TrainState` now stores separate `rng_key`, `sampling_key`, and `balancer_key` streams. The Trainer restores state
before constructing DataModule sources and folds sampling with the restored global step.

This layout change raises the checkpoint schema to version 2. PhiJAX 0.1 checkpoints remain usable with PhiJAX 0.1;
PhiJAX 0.2 reports them as incompatible instead of attempting a partial restore.

## Adaptive balancing

Adaptive balancers now own their update cadence and diagnostic settings:

```python
balancer = GradNormBalancer(
    module.loss_names,
    update_every_n_steps=100,
)

result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=0,
    balancer=balancer,
)
```

Set `update_start_step=0` to update before the first optimizer step. When omitted, the first update occurs after one
complete interval.

`BalancerUpdateConfig` and `BalancerUpdateSchedule` have been removed. Advanced code can place one complete
`BalancerUpdatePlan` directly in `TrainingPlan.balancer_update` for `fit_state()`.
