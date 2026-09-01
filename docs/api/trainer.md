# Trainer

[`Trainer`](#phijax.training.Trainer) owns host-side orchestration. It prepares DataModule sources, places data, calls
one compiled update per step, dispatches hooks, logs metrics, saves checkpoints, handles signals, and releases task
resources.

```python
from phijax import Trainer

trainer = Trainer(
    max_steps=10_000,
    accelerator="auto",
    devices=1,
    precision="32-true",
    logger=True,
)
```

## Common options

| Argument               | Supported values                                                                   | Default     |
| ---------------------- | ---------------------------------------------------------------------------------- | ----------- |
| `accelerator`          | `"auto"`, `"cpu"`, `"gpu"`, `"tpu"`                                                | `"auto"`    |
| `devices`              | Positive count, device-index sequence, or `"auto"`                                 | `1`         |
| `precision`            | `"64-true"`, `"32-true"`, `"16-true"`, `"bf16-true"`, `"16-mixed"`, `"bf16-mixed"` | `"32-true"` |
| `matmul_precision`     | `None`, `"default"`, `"high"`, `"highest"`                                         | `None`      |
| `logger`               | `True`, `False`, `None`, one logger, or an iterable of loggers                     | `True`      |
| `enable_progress_bar`  | Boolean                                                                            | `True`      |
| `enable_model_summary` | Boolean                                                                            | `True`      |

An explicit `strategy` replaces `accelerator` and `devices`. `logger=True` selects TensorBoard when it is installed
and CSV otherwise. See [Loggers](loggers.md), [Callbacks](callbacks.md), and [Precision](precision.md) for related
options.

## Resource ownership

Each `fit()` and `predict()` call releases DataModule, callback, logger, and checkpoint resources after success,
interruption, or failure. A Trainer can therefore be reused without an explicit `close()` call. `close()` and context
manager support remain available for custom integrations that open resources outside a Trainer task.

Environment information is printed when the Trainer is created and only on global rank zero.

## Public API

::: phijax.training.Trainer
    options:
      members:
        - print_environment_info
        - initialize_state
        - compile_train_step
        - prepare_batch_source
        - prepare_batch
        - fit
        - fit_state
        - predict
        - predict_state
        - load_weights
        - latest_checkpoint
        - close
