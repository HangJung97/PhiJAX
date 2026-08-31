# Trainer and functional state

`Trainer` runs on the host. Numerical updates stay in a pure compiled step. Callbacks, module hooks, logging,
checkpoint scheduling, interruption handling, and batch placement stay outside `jax.jit`.
See [Trainer and module hooks](hooks.md) for the complete fit, prediction, exception, and teardown order.

```text
Host Trainer loop

batch source -> batch placement -> JIT numerical update -> callbacks, logging, checkpoints
                                      |
                                      +-> forward + derivatives + loss + grad + Optax
```

The Trainer copies the restored optimizer step to the host once, then maintains a Python step counter while fitting.
The compiled update retains the authoritative `TrainState.step`, which is checked again when fitting finishes. This
keeps ordinary updates asynchronous. A callback that converts a device array to a Python or NumPy scalar deliberately
synchronizes at that callback's cadence.

Internally, private fit and prediction loops keep iteration control separate from the public Trainer facade. Focused
logger and signal connectors own their corresponding host state. This layout follows Lightning's separation of
responsibilities without adding validation, epoch, or optimizer loops that do not match PhiJAX's step-based runtime.

## State and steps

`TrainState` contains model, Optax, balancer, three independent PRNG streams, step, and mixed-precision loss-scaling
state. The streams cover stochastic model execution, DataModule sampling, and adaptive-balancer diagnostics. A complete
checkpoint resumes both the next optimizer update and its deterministic batch sequence.

::: phijax.training.TrainState

::: phijax.training.initialize_train_state

::: phijax.training.make_train_step

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
    enable_progress_bar=True,
    enable_model_summary=True,
    logger=True,
    default_root_dir=".",
)
```

The default display uses TQDM and shows total loss, individual losses, and balancer weights. A plain model summary is
printed before the first batch. Explicit Rich callbacks replace these defaults. See [Callbacks](callbacks.md) for
display customization and [Experiment loggers](loggers.md) for versioned local logging.

## Common fit and prediction

`fit()` accepts an unbound `PhiModule`, DataModule, optimizer, and root seed. It initializes and binds the model,
creates equal static loss weights by default, splits independent PRNG streams, infers objective batch keys, compiles a
`TrainingPlan`, and initializes `TrainState`.

```python
result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=0,
)
predictions = trainer.predict(result, datamodule=data_module)
```

The seed may be a Python integer or an unbatched JAX key. PhiJAX does not modify Python or NumPy global random state.
It splits the root in the stable order `model`, `runtime`, `sampling`, `balancer`. `FitResult.module` is the bound module
used during training; the supplied blueprint remains unmodified. The
[randomness and reproducibility guide](../guides/reproducibility.md) explains stream ownership, deterministic batch
derivation, and checkpoint continuation.

Pass a custom static or adaptive balancer through `balancer`. Adaptive balancers own their update cadence and
diagnostic sampling settings. Their first update occurs after one interval unless `update_start_step` is set:

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

Set `ckpt_path` and, optionally, `ckpt_step` to resume a complete training state. Use `weights_only=True` for transfer
learning. `ckpt_path="last"` restores the latest checkpoint owned by the configured `ModelCheckpoint`. Prediction is
skipped when `predict_batch_source()` returns `None`. The Trainer owns DataModule source access and teardown.

## Explicit state API

Use `fit_state()` when an application needs a custom bound `BasePhiModule`, explicit `TrainState`, `TrainingPlan`,
compiled step, or batch source. Use `predict_state()` for checkpoint templates and custom state:

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

A custom `TrainStep` must increment `TrainState.step` exactly once. The Trainer uses its corresponding host counter for
batch-source indices, callback contexts, adaptive-balancer cadence, logging, and checkpoint paths.

`TrainState.sampling_key` and `TrainState.balancer_key` replace the separate 0.1 method arguments. Restoration occurs
before the Trainer constructs a DataModule source. The source folds its restored sampling key with the global step, so
resumed training continues the same deterministic sequence.

Each task closes its callback and checkpoint resources, even after an error. You can reuse one Trainer across `fit()`
and `predict()` calls without calling `close()`. The idempotent `close()` method and context manager remain available
for integrations that open resources outside these tasks.

After `Ctrl+C`, fitting returns the last completed state with `FitResult.interrupted=True`. The caller can then run
prediction. `SIGTERM` runs exception hooks and teardown, waits for checkpoint cleanup, and exits with the standard
signal status.

::: phijax.training.FitResult

::: phijax.training.TrainingPlan

::: phijax.training.Trainer

options:
members:
\- print_environment_info
\- initialize_state
\- compile_train_step
\- prepare_batch_source
\- prepare_batch
\- fit
\- fit_state
\- predict
\- predict_state
\- load_weights
\- latest_checkpoint
\- close

::: phijax.training.build_training_plan

## Structured diagnostics and host-side logging

`BasePhiModule.training_step()` may return a loss mapping or `TrainingOutput`. Diagnostics remain part of the static
compiled PyTree but do not affect gradients unless the loss explicitly consumes them. Use them for inexpensive values
that are already available during objective evaluation, such as residual statistics.

```python
from phijax import TrainingOutput

return TrainingOutput(
    losses={"pde/heat": heat_loss},
    diagnostics={
        "residual/mean_abs": jnp.mean(jnp.abs(residuals)),
        "residual/by_region": regional_residuals,
    },
)
```

Diagnostic names and PyTree structure must remain fixed across compiled calls. Scalar diagnostics are logged by
default. Array diagnostics remain available to callbacks but are not implicitly reduced or transferred.

Use the module's host-side batch-end hook to expose an important scalar in the progress bar:

```python
def on_train_batch_end(self, model_state, context):
    self.log("train/residual/mean_abs", context.metrics["train/residual/mean_abs"], prog_bar=True)
    return model_state, context.metrics
```

Use `TrainingOutput` for inexpensive diagnostics already produced while evaluating the differentiated objective.
Expensive optional diagnostics, such as full-grid residual statistics or additional gradient norms, should remain in a
separately compiled evaluator and run only at the required reporting cadence.

Diagnostic keys, shapes, and dtypes must stay fixed across compiled calls. Reduce large arrays on the device before
logging them. The Trainer reports diagnostics but does not interpret them or use them to change gradients.

::: phijax.TrainingOutput

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

`DataParallelStrategy` is experimental DDP-style synchronous data parallelism, not full Lightning DDP parity. Call
`initialize_distributed()` before JAX backend initialization for multi-process use. Every process must execute the
same compiled steps and collectives in the same order. State is replicated, leading batch axes are sharded, metrics
are reduced by JAX computations, and checkpoint and logger side effects remain global-rank-zero operations. Ordinary
CI covers single-process CPU behavior; validate a target multi-host environment before long runs.

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

## Evaluation policy

PhiJAX does not provide a generic validation loop. PINN evaluation may require dense reference grids, conserved
quantities, experimental measurements, or application-specific sampling schedules. Applications should run these
checks separately through `Trainer.predict()` and the evaluation APIs rather than treating supervised validation
batches as a universal training-stage contract.
