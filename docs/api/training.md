# Trainer and functional state

`Trainer` runs on the host. Numerical updates stay in a pure compiled step. Callbacks, module hooks, logging,
checkpoint scheduling, interruption handling, and batch placement stay outside `jax.jit`.
See [Trainer and module hooks](hooks.md) for the complete fit, prediction, exception, and teardown order.

## State and steps

`TrainState` contains model, Optax, balancer, PRNG, step, and mixed-precision loss-scaling state. A complete checkpoint
can resume both the next optimizer update and the deterministic batch sequence.

::: phijax.training.TrainState

::: phijax.training.initialize_train_state

::: phijax.training.make_train_step

::: phijax.training.with_balancer_updates

## Trainer

```python
from phijax import Trainer

trainer = Trainer(
    max_steps=10_000,
    accelerator="auto",
    devices=1,
    precision="32-true",
    matmul_precision=None,
    deterministic=True,
    log_every_n_steps=10,
)
```

`fit` accepts a `TrainingPlan`. It asks the DataModule for a step-indexed source that provides the plan's batch keys.
Pass `sampling_key` for deterministic batches. Pass `balancer_key` when using fixed NTK or gradient-norm diagnostic
batches. Advanced users may instead supply a compiled step and an iterable or callable source.

Set `ckpt_path` and, optionally, `ckpt_step` to resume a complete training state. Use `weights_only=True` for transfer
learning. `predict` accepts a fresh or in-memory `TrainState` and loads model weights when `ckpt_path` is set. If no
batches are passed, it asks the DataModule for `predict_batch_source()`. Prediction is skipped when that method returns
`None`. The Trainer owns DataModule source access and stage teardown during both tasks.

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

Each task closes its callback and checkpoint resources, even after an error. You can reuse one Trainer across `fit()`
and `predict()` calls without calling `close()`. The idempotent `close()` method and context manager remain available
for integrations that open resources outside these tasks.

After `Ctrl+C`, `fit` returns the last completed state with `FitResult.interrupted=True`. The caller can then run
prediction. `SIGTERM` runs exception hooks and teardown, waits for checkpoint cleanup, and exits with the standard
signal status.

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

`matmul_precision` controls JAX dot and convolution arithmetic during fit and prediction tracing. Use `default`,
`high`, or `highest`. Set it to `None` to preserve the current JAX policy. The Trainer restores the previous policy
after each task. Therefore, an explicit Trainer value temporarily overrides `JAX_DEFAULT_MATMUL_PRECISION`, while
`None` preserves the environment value. See JAX's
[matrix-multiplication precision guide](https://docs.jax.dev/en/latest/201/precision.html) for the meaning of each
level. Matmul precision controls dot-product arithmetic and does not change parameter or activation dtypes.

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

`Trainer` accepts constructed Python objects and never reads Hydra configuration. Optional factories live under
`phijax.integrations.hydra`; see [Configuration integrations](configuration.md). This keeps the Trainer usable from
plain Python and lets each application own its config tree and entrypoints.
