# Train and predict

This guide explains the normal training workflow and when to use the explicit state API.

## Run the common workflow

Create a module blueprint, DataModule, Optax optimizer, and Trainer. The Trainer binds the model after the DataModule
has prepared its input statistics.

```python
trainer = Trainer(max_steps=10_000, accelerator="auto")

result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=0,
)
predictions = trainer.predict(result, datamodule=data_module)
```

`fit()` performs the following work:

1. Set up the DataModule and read its input statistics.
2. Initialize and bind the model factory.
3. Create equal static loss weights unless a balancer is supplied.
4. Split independent runtime, sampling, and balancer PRNG streams.
5. Infer objective batch keys and build the compiled training plan.
6. Initialize or restore `TrainState`.
7. Run host-side sampling around one JIT-compiled update per step.

The original module blueprint is unchanged. Use `result.module` for the bound module and `result.state` for the final
functional state.

## Choose a loss balancer

Pass a static or adaptive balancer directly to `fit()`. Adaptive balancers own their update interval and diagnostic
sampling settings.

```python
from phijax.balancers import GradNormBalancer

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

See [Loss balancers](balancers.md) for update timing, diagnostic batches, and logged metrics.

## Resume or load weights

Pass an explicit checkpoint path to restore a complete training state. Add `ckpt_step` to select one saved step. With
a configured `ModelCheckpoint`, `ckpt_path="last"` selects its latest checkpoint.

```python
result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=0,
    ckpt_path="last",
)
```

Use `weights_only=True` when the model parameters should be restored into fresh optimizer, balancer, PRNG, and step
state. Complete restoration resumes the deterministic sampling sequence.

## Use the explicit state API

Use `fit_state()` when an application supplies a bound `BasePhiModule`, explicit `TrainState`, custom `TrainingPlan`,
or custom batch source. Use `predict_state()` when restoring into an explicit state template.

```python
result = trainer.fit_state(
    bound_module,
    training_plan,
    initial_state,
    datamodule=data_module,
)
predictions = trainer.predict_state(
    result.module,
    result.state,
    datamodule=data_module,
)
```

A custom compiled step must increment `TrainState.step` exactly once. The Trainer uses a matching host counter for
batch indices, callbacks, logging, adaptive-balancer updates, and checkpoints.

## Understand the execution boundary

```text
Host: sample batch -> place batch -> call compiled update -> hooks -> log/checkpoint
                                      |
Device:                         loss -> gradient -> Optax update
```

Sampling, hooks, logging, checkpoint scheduling, and signal handling stay in Python. The loss, derivatives, gradient,
balancing, and Optax update run inside one JIT-compiled function. The Trainer reads the restored step once before the
loop and verifies it once at the end. Converting a device value to a Python or NumPy scalar from a callback introduces
an intentional synchronization at that callback's cadence.

## Report extra diagnostics

Return `TrainingOutput` when the compiled objective already has useful fixed-shape diagnostics.

```python
return TrainingOutput(
    losses={"pde/heat": heat_loss},
    diagnostics={
        "residual/mean_abs": jnp.mean(jnp.abs(residuals)),
        "residual/by_region": regional_residuals,
    },
)
```

Scalar diagnostics are logged automatically. Arrays remain available to callbacks and are not reduced implicitly.
Use `self.log()` from `on_train_batch_end()` to change where a scalar appears. Run expensive reporting-only
calculations in a separate compiled evaluator at the required cadence.

## Handle interruption and evaluation

After `Ctrl+C`, `fit()` returns the last completed state with `result.interrupted=True`. The caller may then predict.
`SIGTERM` runs exception hooks and cleanup before exiting with the standard signal status.

Prediction is skipped when `predict_batch_source()` returns `None`. PhiJAX does not impose a generic validation loop;
use prediction and the evaluation APIs for application-specific reference grids, measurements, or physical checks.

## Next steps

- [Trainer reference](../api/trainer.md)
- [Training state and plans](../api/training.md)
- [Callbacks](../api/callbacks.md)
- [Checkpointing](../api/checkpointing.md)
- [Randomness and reproducibility](reproducibility.md)
