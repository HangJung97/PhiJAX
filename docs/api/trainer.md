# Trainer

`Trainer` runs the host side of fitting and prediction. It prepares data sources, places arrays, calls one compiled
update per step, dispatches hooks, logs metrics, saves checkpoints, handles interruption, and releases task resources.

## Basic use

Construct a Trainer, fit a [`PhiModule`](module.md#phijax.core.PhiModule) blueprint, and pass the returned
[`FitResult`](training.md#phijax.training.FitResult) to `predict()`:

```python
from phijax import Trainer

trainer = Trainer(max_steps=10_000, accelerator="auto")
result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=0,
)
predictions = trainer.predict(result, datamodule=data_module)
```

The Trainer initializes the model and functional state during `fit()`. Applications do not need to call `close()`
after a normal fit or prediction task. See [Train and predict](../guides/training.md) for checkpoint restoration,
adaptive loss balancing, interruption, and the explicit state API.

## What the Trainer manages

The Trainer owns orchestration around the compiled numerical update:

```text
DataModule source -> cast and place batch -> compiled update -> hooks -> logging and checkpointing
                                              |
                                              +-> explicit TrainState
```

It manages:

- DataModule setup, named batch construction, prediction batches, and teardown;
- model initialization and independent runtime, sampling, and balancer PRNG streams;
- precision conversion and device or data-parallel placement;
- callback and module lifecycle ordering;
- metric routing, progress displays, model summaries, and experiment loggers;
- checkpoint restoration and persistence; and
- graceful interruption and resource cleanup.

Losses, derivatives, gradients, balancing, and Optax updates remain inside the JIT-compiled step. Python hooks and
filesystem operations remain outside it.

## Configuration

### Execution

| Argument            | Supported values                                      | Default  |
| ------------------- | ----------------------------------------------------- | -------- |
| `max_steps`         | Positive integer                                      | Required |
| `deterministic`     | Boolean reproducibility declaration                   | `True`   |
| `callbacks`         | Ordered iterable of callback instances                | `()`     |
| `compilation_cache` | Mapping with `enabled` and, when enabled, `directory` | `None`   |

`max_steps` counts optimizer updates within one fit call. A custom step passed through `fit_state()` must increment
`TrainState.step` exactly once per completed update.

### Devices and precision

| Argument                     | Supported values                                                                   | Default     |
| ---------------------------- | ---------------------------------------------------------------------------------- | ----------- |
| `accelerator`                | `"auto"`, `"cpu"`, `"gpu"`, `"tpu"`                                                | `"auto"`    |
| `devices`                    | Positive count, device-index sequence, or `"auto"`                                 | `1`         |
| `strategy`                   | A configured `Strategy`, which replaces `accelerator` and `devices`                | `None`      |
| `precision`                  | `"64-true"`, `"32-true"`, `"16-true"`, `"bf16-true"`, `"16-mixed"`, `"bf16-mixed"` | `"32-true"` |
| `matmul_precision`           | `None`, `"default"`, `"high"`, `"highest"`                                         | `None`      |
| `derivative_dtype`           | Floating dtype override                                                            | `None`      |
| `initial_loss_scale`         | Positive FP16 dynamic-loss scale                                                   | `32768.0`   |
| `loss_scale_growth_interval` | Positive interval between finite FP16 scale increases                              | `2000`      |

An explicit [`Strategy`](strategies.md) controls placement and distributed behavior. See [Precision](precision.md) for
dtype behavior and supported device combinations.

### Logging and display

| Argument               | Supported values                                               | Default |
| ---------------------- | -------------------------------------------------------------- | ------- |
| `logger`               | `True`, `False`, `None`, one logger, or an iterable of loggers | `True`  |
| `default_root_dir`     | Parent directory for default versioned logs                    | `"."`   |
| `log_every_n_steps`    | Positive scalar-logging interval                               | `10`    |
| `enable_progress_bar`  | Boolean                                                        | `True`  |
| `enable_model_summary` | Boolean                                                        | `True`  |

`logger=True` selects TensorBoard when it is installed and CSV otherwise. The Trainer adds a TQDM progress bar and a
plain model summary unless callbacks provide replacements. See [Loggers](loggers.md) and [Callbacks](callbacks.md) for
backend and display options.

## Common and advanced workflows

Use `fit()` and `predict()` for normal application code. The Trainer initializes and binds the model, constructs the
training plan, creates state, and obtains batches from the DataModule.

Use `fit_state()` and `predict_state()` when an integration supplies a bound `BasePhiModule`, explicit `TrainState`,
custom `TrainingPlan`, or custom batch source. Supporting methods such as `initialize_state()`, `compile_train_step()`,
`prepare_batch_source()`, and `prepare_batch()` expose the same runtime policies without creating another training
loop.

Checkpoint-enabled applications can use `ckpt_path="last"` with `fit()`, call `latest_checkpoint()` explicitly, or
load model parameters into an existing state with `load_weights()`.

## Metrics and status

The Trainer exposes the latest metric views without attaching mutable Trainer state to a module:

| Property               | Contents                                              |
| ---------------------- | ----------------------------------------------------- |
| `callback_metrics`     | Complete merged metric mapping available to callbacks |
| `logged_metrics`       | Scalar metrics selected for experiment loggers        |
| `progress_bar_metrics` | Scalar metrics selected for progress displays         |
| `interrupted`          | Whether the latest fit stopped after an interrupt     |
| `received_sigterm`     | Whether the latest fit received `SIGTERM`             |
| `logger`               | Configured logger collection                          |

Metric mappings contain device values until a logger, display, checkpoint monitor, or caller requires host
conversion. See [Logging and monitoring](../guides/logging.md) for module and callback metric routing.

## Cleanup and interruption

Each `fit()` and `predict()` call releases DataModule, callback, logger, and checkpoint resources after success,
interruption, or failure. A Trainer can be reused without an explicit `close()` call. The idempotent `close()` method
and context-manager support remain available for integrations that open resources outside a Trainer task.

The first `Ctrl+C` returns the last completed state with `FitResult.interrupted=True`. `SIGTERM` runs exception hooks
and cleanup before exiting with the standard signal status. Environment information is printed when the Trainer is
created and only on global rank zero.

## API reference

::: phijax.training.Trainer
    options:
      members:
        - logger
        - interrupted
        - received_sigterm
        - logged_metrics
        - progress_bar_metrics
        - callback_metrics
        - fit
        - predict
        - fit_state
        - predict_state
        - initialize_state
        - compile_train_step
        - prepare_batch_source
        - prepare_batch
        - load_weights
        - latest_checkpoint
        - callback_state_dict
        - load_callback_state_dict
        - print_environment_info
        - close
