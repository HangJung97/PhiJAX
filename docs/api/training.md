# Training state and plans

This page documents the explicit state and compiled-step contracts. Most applications can use
[`Trainer.fit()`](trainer.md#phijax.training.Trainer.fit) and do not need to construct these objects directly. See the
[training guide](../guides/training.md) for the common workflow and the [Trainer reference](trainer.md) for runtime
options.

## State

`TrainState` contains model parameters, Optax state, balancer state, independent PRNG streams, the optimizer step, and
mixed-precision loss-scaling state. A complete checkpoint restores every field.

::: phijax.training.TrainState

::: phijax.training.initialize_train_state

## Compiled update

`make_train_step()` creates one JIT-compiled numerical update. It evaluates the objective, combines its losses,
computes gradients, applies the Optax transformation, and returns the updated state and metrics.

::: phijax.training.make_train_step

## Fit results and plans

`FitResult` returns the bound module and final functional state. `TrainingPlan` describes an explicit compiled update,
loss balancer, and optional adaptive-balancer update. These contracts power `Trainer.fit_state()`.

::: phijax.training.FitResult

::: phijax.training.TrainingPlan

::: phijax.training.build_training_plan

## Training output

`TrainingOutput` carries named scalar losses and optional fixed-structure diagnostics from the compiled step. Scalar
diagnostics are logged by default. Array diagnostics remain available to callbacks and are never reduced implicitly.

::: phijax.TrainingOutput
