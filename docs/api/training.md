# Trainer and functional state

`Trainer` is host-side orchestration. Numerical updates remain in a pure compiled train step, while callbacks, module
hooks, logging, checkpoint scheduling, graceful interruption, and batch placement remain outside `jax.jit`.

## State and steps

`TrainState` contains model, Optax, balancer, PRNG, step, and mixed-precision loss-scaling state. A complete checkpoint
can therefore resume the next optimizer update and deterministic batch draw.

::: phijax.training.TrainState

::: phijax.training.initialize_train_state

::: phijax.training.make_train_step

::: phijax.training.with_balancer_updates

## Trainer

```yaml
_target_: phijax.training.Trainer
max_steps: 10000
accelerator: auto
devices: 1
precision: 32-true
matmul_precision: null
deterministic: true
log_every_n_steps: 10
```

`fit` accepts a `TrainingPlan` and asks the supplied DataModule for its step-indexed source using the plan's batch keys.
Pass `sampling_key` for deterministic training batches and `balancer_key` when fixed NTK or gradient-norm diagnostics
are configured. A raw compiled step plus an explicit iterable or callable remains available for programmatic use.
Pass an optional `ckpt_path` and `ckpt_step` directly to the trainer for full-state resumption, or set
`weights_only=True` for transfer learning. `predict` accepts the same fresh or in-memory `TrainState` shape and restores
model weights when `ckpt_path` is set. When batches are omitted, it requests `predict_batch_source()` from the supplied
DataModule and skips prediction if the source is `None`. Neither lifecycle requires callers to invoke the low-level
checkpoint backend. Passing the DataModule through `datamodule` transfers source and stage-teardown ownership to the
trainer.

```python
fit_result = trainer.fit(
    module,
    training_plan,
    initial_state,
    datamodule=data_module,
    sampling_key=sampling_key,
    balancer_key=balancer_key,
    ckpt_path=checkpoint_directory,
)
predictions = trainer.predict(
    module,
    fit_result.state,
    datamodule=data_module,
)
```

A graceful `Ctrl+C` returns the last completed functional state with `FitResult.interrupted=True`, allowing a caller
to continue into prediction. `SIGTERM` runs exception hooks and teardown, waits for checkpoint cleanup, and then
terminates the process with the conventional signal exit status.

::: phijax.training.FitResult

::: phijax.training.TrainingPlan

::: phijax.training.BalancerUpdateSchedule

::: phijax.training.Trainer
options:
members:
\- print_environment_info
\- initialize_state
\- compile_train_step
\- prepare_batch_source
\- prepare_batch
\- fit
\- resume_latest
\- predict
\- load_weights
\- latest_checkpoint
\- close

## Precision

| Mode         | Parameter dtype | Compute dtype | Output dtype | Dynamic loss scaling |
| ------------ | --------------- | ------------- | ------------ | -------------------- |
| `64-true`    | float64         | float64       | float64      | No                   |
| `32-true`    | float32         | float32       | float32      | No                   |
| `16-true`    | float16         | float16       | float16      | No                   |
| `bf16-true`  | bfloat16        | bfloat16      | bfloat16     | No                   |
| `16-mixed`   | float32         | float16       | float32      | Yes                  |
| `bf16-mixed` | float32         | bfloat16      | float32      | No                   |

`initial_loss_scale` applies only to `16-mixed`. The loss is scaled before differentiation and gradients are unscaled
before the Optax update. Non-finite gradients skip the update and reduce the scale; sustained finite updates allow it
to grow. BFloat16 normally does not require this because it retains a float32-like exponent range.

`matmul_precision` independently controls JAX dot and convolution arithmetic during lazy fit and prediction tracing.
Use `default`, `high`, or `highest`; `null` preserves an externally configured JAX policy. The trainer restores the
previous process policy after each lifecycle, so experiments do not leak overrides into later tasks.

::: phijax.training.PrecisionPolicy

::: phijax.training.configure_precision

## Strategies

`SingleDeviceStrategy` places complete state and batches on one selected backend device. `DataParallelStrategy`
replicates state and shards non-scalar batch leaves over their leading dimension. Batch sizes must be divisible by the
number of local shards.

Distributed initialization must occur before any operation that initializes a JAX backend.

::: phijax.training.Strategy

::: phijax.training.SingleDeviceStrategy

::: phijax.training.DataParallelStrategy

::: phijax.training.create_strategy

::: phijax.training.initialize_distributed

## Configuration boundary

`Trainer` accepts constructed Python objects and never reads Hydra configuration. Optional factories and assembly
helpers live under `phijax.integrations.hydra`; see [Configuration integrations](configuration.md). This keeps the core
runtime usable from plain Python and lets each project own its config tree and executable entrypoints.
