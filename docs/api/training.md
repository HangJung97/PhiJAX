# Training state and results

This page documents the functional state and fit-result contracts. Most applications can use
[`Trainer.fit()`](trainer.md#phijax.training.Trainer.fit) and do not need to construct these objects directly. See the
[training guide](../guides/training.md) for the common workflow and the [Trainer reference](trainer.md) for runtime
options.

## State

`TrainState` contains model parameters, Optax state, balancer state, independent PRNG streams, the optimizer step,
mixed-precision loss-scaling state, and the model's input normalization statistics. A complete checkpoint restores every
field. The statistics are never differentiated; see
[input normalization statistics](checkpointing.md#input-normalization-statistics).

::: phijax.training.TrainState

## Fit results

`FitResult` returns the bound module and final functional state. Pass it to `Trainer.predict()` to predict in memory.

::: phijax.training.FitResult

## Training output

`TrainingOutput` carries named scalar losses and optional fixed-structure diagnostics from the compiled step. Scalar
diagnostics are logged by default. Array diagnostics remain available to callbacks and are never reduced implicitly.

::: phijax.TrainingOutput
