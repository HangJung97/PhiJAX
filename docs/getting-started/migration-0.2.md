# Migrating from 0.1 to 0.2

PhiJAX 0.2 makes `Trainer.fit()` the common application API. It initializes the model and functional state, infers
batch routing, compiles the update, and owns DataModule sources. The explicit 0.1 execution path remains available
under names that describe its state-level contract.

## API replacements

| PhiJAX 0.1                                                       | PhiJAX 0.2                                                             |
| ---------------------------------------------------------------- | ---------------------------------------------------------------------- |
| `PhiModule(model_apply, objective, model_summary=...)`           | `PhiModule(model_factory, objective)`                                  |
| `initialize_state()` + `compile_train_step()` + `TrainingPlan()` | `Trainer.fit(..., optimizer=..., seed=...)`                            |
| `Trainer.fit(module, plan, state, ...)`                          | `Trainer.fit_state(module, plan, state, ...)`                          |
| `Trainer.predict(module, state, ...)`                            | `Trainer.predict(fit_result, ...)` or `predict(module, ckpt_path=...)` |
| Explicit post-fit module and state                               | `Trainer.predict(fit_result, ...)`                                     |
| `resume_latest(...)`                                             | `fit(..., ckpt_path="last")`                                           |
| `MetricRoute` and `configure_metric_routes()`                    | `self.log(..., logger=..., prog_bar=...)` in `on_train_batch_end()`    |
| `from phijax.module import PhiModule`                            | `from phijax import PhiModule`                                         |
| `from phijax.module import BasePhiModule`                        | `from phijax.core import BasePhiModule`                                |
| `from phijax.training.trainer import FitResult`                  | `from phijax.training import FitResult`                                |
| `trainer.set_logger(instantiate_loggers(cfg.logger, trainer))`   | `instantiate_trainer(cfg.trainer, callbacks, logger=loggers)`          |
| `batch["mask"]` in prediction batches                            | `batch["valid_predict_mask"]`                                          |

Top-level imports are the recommended application API. PhiJAX 0.2 removes the old `phijax.module` path; custom module
implementations that need a deep import should use `phijax.core`.

Configured loggers are now constructed before the Trainer and passed to its constructor. `instantiate_loggers()` no
longer accepts a Trainer, and `Trainer.set_logger()` has been removed. Built-in logger resources start lazily when the
Trainer begins a task. Use `to_hyperparameters(cfg)` when a Hydra project wants to pass its composed configuration to
`Trainer.fit(..., hyperparameters=...)`.

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

### Stored input statistics

From PhiJAX 0.2.0b6, `TrainState.input_statistics` stores the model's input normalization statistics, and checkpoints
use schema version 3. A full resume and prediction from a checkpoint use the stored statistics; transfer learning with
`weights_only=True` uses the current DataModule's statistics. Schema-2 checkpoints from 0.2.0b5 still restore with the
current DataModule's statistics and a warning. PhiJAX 0.2.0b5 cannot read schema-3 checkpoints.

Existing prediction scripts need no change. A restore template built without statistics takes them from the module,
and the Trainer rebinds the module to the checkpoint statistics when they differ.

### Periodic features in physical units

From PhiJAX 0.2.0b6, periodic axes are declared with physical periods and embedded from the physical coordinate.
Input normalization no longer affects them.

| PhiJAX 0.2.0b5 and earlier                                                          | PhiJAX 0.2.0b6 and later                              |
| ----------------------------------------------------------------------------------- | ----------------------------------------------------- |
| `periodic_features_kwargs: {axes: [1], frequencies: [pi]}` with `input_norm: false` | `periodic_features_kwargs: {periods: {1: 2.0}}`       |
| Measured period as that axis's `std`, with frequency `2*pi`                         | `PhiDataModule.input_periods()` returning `{axis: T}` |
| `PeriodicFeatures(input_dim, axes, frequencies)`                                    | `PeriodicFeatures(input_dim, periods={axis: T})`      |
| Model factory `(key, input_mean, input_std, precision)`                             | Also receives `input_periods`                         |
| `bind_input_statistics(statistics)`                                                 | `bind_inputs(statistics, periods)`                    |

Checkpoints trained with frequency-based periodic features apply a different phase and cannot be reused with the new
features.

### Checkpoint retention

From PhiJAX 0.2.0b6, `ModelCheckpoint` owns checkpoint retention, as in Lightning. `OrbaxCheckpointIO` no longer
accepts `max_to_keep`, and `save_top_k` defaults to `1`.

| PhiJAX 0.2.0b5 and earlier                             | PhiJAX 0.2.0b6 and later                                         |
| ------------------------------------------------------ | ---------------------------------------------------------------- |
| `OrbaxCheckpointIO(..., max_to_keep=3)`                | `ModelCheckpoint(..., monitor="step", mode="max", save_top_k=3)` |
| `OrbaxCheckpointIO(..., max_to_keep=None)`             | `ModelCheckpoint(..., save_top_k=-1)`                            |
| `ModelCheckpoint(checkpoint_io)` kept every checkpoint | `ModelCheckpoint(checkpoint_io)` keeps the latest checkpoint     |

In a Hydra config, move the retention count from `checkpoint_io` to the callback:

```yaml
model_checkpoint:
  _target_: phijax.callbacks.ModelCheckpoint
  every_n_steps: 5000
  monitor: step
  mode: max
  save_top_k: 3
  checkpoint_io:
    _target_: phijax.training.OrbaxCheckpointIO
    directory: ${paths.output_dir}/checkpoints
```

### Model-only restore

From PhiJAX 0.2.0b6, weights-only restores read only the model state, so the optimizer and balancer may differ from
the checkpoint's. `CheckpointIO.restore_weights()` is replaced by `restore_model_state()`, which takes a model-state
template instead of a full state.

| PhiJAX 0.2.0b5 and earlier                | PhiJAX 0.2.0b6 and later                                                                   |
| ----------------------------------------- | ------------------------------------------------------------------------------------------ |
| `checkpoint_io.restore_weights(state)`    | `replace(state, model_state=checkpoint_io.restore_model_state(state.model_state))`         |
| Custom `CheckpointIO.restore_weights()`   | Implement `restore_model_state()`, `restore_input_statistics()`, `restore_input_periods()` |
| Weights-only fit with a changed optimizer | Works; only the model must match the checkpoint                                            |

Outside the Trainer, `load_model_state(ckpt_path, state.model_state)` opens the checkpoint, restores the model
state, and closes it.

### Prediction from a checkpoint

From PhiJAX 0.2.0b6, `Trainer.predict()` accepts a module blueprint with `ckpt_path`, and `Trainer.predict_state()` is
removed. A prediction script no longer builds a training-state template, so it needs no optimizer, balancer, or seed.

| PhiJAX 0.2.0b5 and earlier                                                     | PhiJAX 0.2.0b6 and later                         |
| ------------------------------------------------------------------------------ | ------------------------------------------------ |
| `prepare_model()` + `initialize_state()` + `predict_state(..., ckpt_path=...)` | `predict(module, ckpt_path=..., datamodule=...)` |
| `predict_state(fit_result.module, fit_result.state, ...)`                      | `predict(fit_result, ...)`                       |

Checkpoints written by 0.2.0b5 store no input statistics or periods, so prediction takes them from the fit stage of
`datamodule` and warns.

### Global `max_steps`

From PhiJAX 0.2.0b6, `Trainer(max_steps=...)` is the global optimizer step at which training stops, as in Lightning.
Previously, every fit call ran `max_steps` more updates, so a resumed run trained past the configured total.

| PhiJAX 0.2.0b5 and earlier                            | PhiJAX 0.2.0b6 and later                                       |
| ----------------------------------------------------- | -------------------------------------------------------------- |
| Resume from step `N` with `max_steps=extra`           | Resume from step `N` with `max_steps=N + extra`                |
| Logging every `log_every_n_steps` updates of the call | Logging at global steps divisible by `log_every_n_steps`       |
| Progress bars restart at `0` after a resume           | Progress bars start at the restored step                       |
| Resuming a finished run trains `max_steps` more steps | Resuming a finished run runs the fit hooks without training    |
| `RichProgressBar(total=max_steps)`                    | `RichProgressBar()`, which counts to the Trainer's `max_steps` |

### Early stopping

From PhiJAX 0.2.0b6, `EarlyStopping` follows Lightning's callback and checks metrics in `on_train_metrics()`.

| PhiJAX 0.2.0b5 and earlier                            | PhiJAX 0.2.0b6 and later                                       |
| ----------------------------------------------------- | -------------------------------------------------------------- |
| Stops after `patience + 1` checks without improvement | Stops after `patience` checks without improvement              |
| `best` and `bad_checks` attributes                    | `best_score` and `wait_count` attributes                       |
| A NaN loss counts as one non-improving check          | A NaN or infinite loss stops immediately (`check_finite=True`) |
| Sees only metrics from the compiled step              | Also sees module-logged and callback metrics                   |
| Progress restarts on resume                           | Progress is stored in checkpoints                              |
| Callbacks stop by returning `True`                    | Callbacks set `trainer.should_stop = True`; `True` still works |

The new rule stops one check earlier. To stop at the same step as before, increase `patience` by one.

## Per-device batch sizes

From PhiJAX 0.2.0b6, training batch sizes, NTK diagnostic sizes, and prediction chunk sizes count rows per device, as
in Lightning. Runs on one device are unchanged.

| Size                                 | PhiJAX 0.2.0b5 and earlier               | PhiJAX 0.2.0b6 and later          |
| ------------------------------------ | ---------------------------------------- | --------------------------------- |
| `NamedBatchSource` batch sizes       | Rows per process, split over its devices | Rows per device                   |
| `ExactNTKBalancer(kernel_size=...)`  | Rows per process, split over its devices | Rows per device                   |
| `ChunkedPredictionSource` chunk size | Rows per chunk, split over every device  | Rows per device in each chunk     |
| Divisibility by the device count     | Required for training and NTK sizes      | Not required for built-in sources |

To keep the same global batch on `n` devices, divide each integer size by `n`. The `"all"` policy keeps its meaning.

## Optional Hydra and IO extras

From PhiJAX 0.2.0b6, Hydra and MATLAB or HDF5 support are optional extras. Add them to projects that use these features:

| Feature                                                                    | Install         |
| -------------------------------------------------------------------------- | --------------- |
| `phijax.integrations`, `task_wrapper()`, `extras()`, `pre_hydra_routine()` | `phijax[hydra]` |
| `.mat` and HDF5 files in `load_arrays()`, `save_mat=True` artifacts        | `phijax[io]`    |

Hydra-based projects, such as those built from the template, usually need `phijax[hydra,io]`. Without the extra, these
features raise `ModuleNotFoundError` naming the extra to install.

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
