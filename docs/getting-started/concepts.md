# Core concepts

PhiJAX separates differentiable calculations from the Python code that runs an experiment. JAX transformations stay
visible, while the Trainer handles repeated training, prediction, logging, and cleanup tasks.

## From an equation to an optimizer update

```text
equation residuals
       |
       v
ResidualTerm ----> named scalar losses ----> LossBalancer ----> total loss
       |
       v
CompositeObjective ----> bound PhiModule ----> TrainingPlan ----> Trainer.fit_state()
```

For the common API, `Trainer.fit()` creates the `TrainingPlan` internally. The diagram shows the resolved numerical
path that advanced users can construct directly with `fit_state()`.

The pieces have distinct responsibilities:

| Object               | Responsibility                                                                |
| -------------------- | ----------------------------------------------------------------------------- |
| equation callable    | Evaluate physical residual arrays from a model, explicit state, and one batch |
| `ResidualTerm`       | Reduce residual arrays into stable, named scalar losses                       |
| `CompositeObjective` | Combine independent initial, boundary, data, and PDE terms                    |
| `PhiModule`          | Describe a model factory and its unweighted objective                         |
| `LossBalancer`       | Combine named losses and expose weight diagnostics                            |
| `TrainingPlan`       | Advanced contract binding a compiled update to its required batches           |
| `Trainer`            | Run lifecycle hooks, place batches, restore checkpoints, log, and clean up    |

The optimizer and balancer remain outside `PhiModule`. You can change either one without creating a new module or
rewriting an equation.

Each Trainer task handles its own setup and teardown. It cleans up DataModules, callbacks, loggers, and checkpoint
managers after success, interruption, or failure. Applications can call `fit()` and `predict()` without calling
`close()`.

## Explicit model and training state

A `ModelFactory` receives a model key, optional normalization statistics, and the Trainer precision policy. It returns
an `InitializedModel` containing three values:

- `apply`, a pure callable mapping `(model_state, point)` to a prediction;
- `state`, the differentiable parameter and variable PyTree; and
- `summary`, an optional architecture-summary callable.

`TrainState` adds optimizer state, loss-balancer state, independent model-runtime, sampling, and balancer PRNG keys,
the global step, and precision state. This explicit state can move between devices and checkpoints without hidden
mutable model attributes.

`PhiModule` begins as an uninitialized blueprint. `Trainer.fit()` binds its factory to a separate module instance and
returns that instance as `FitResult.module`; it does not mutate the blueprint.

## Data vocabulary

The data API distinguishes storage from sampling:

| Term            | Meaning                                                                                |
| --------------- | -------------------------------------------------------------------------------------- |
| `HostPool`      | Immutable NumPy arrays and reconstruction metadata stored on the CPU                   |
| sampler         | Policy for selecting finite rows or generating fresh coordinates from explicit keys    |
| batch source    | Step-indexed collection of named samplers, or a finite sequence of prediction chunks   |
| `PhiDataModule` | Application-owned lifecycle that constructs pools and exposes training/prediction data |

The DataModule does not choose a device. The Trainer prepares sampler state and places each batch immediately before
the compiled update. Input normalization is opt-in: override `input_statistics()` when a model should receive a mean
and standard deviation; the base DataModule returns `None`.

## Reproducible randomness

PhiJAX does not hide JAX random keys. Model initialization, training state, sampling, and adaptive-balancer diagnostics
receive separate keys. `Trainer.fit()` splits its root in the stable order `model`, `runtime`, `sampling`, `balancer`.
A named training source folds its key with the restored global step, so resumed training continues the same sampling
sequence. See [Randomness and reproducibility](../guides/reproducibility.md) for stream ownership, checkpoint behavior,
and the limits of cross-platform determinism.

## Where Hydra fits

The core runtime accepts constructed Python objects and does not read configuration. Hydra projects can use
`phijax.integrations.hydra` to build those objects from their own YAML files. This configuration layer does not change
the numerical APIs described above.

Continue with the [heat-equation quickstart](quickstart.md), then use the task-oriented guides to replace each example
component with application-specific behavior.
