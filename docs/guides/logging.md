# Log and monitor training

PhiJAX logs scalar training metrics on the host. The compiled update returns values; module hooks and callbacks decide
where those values are displayed, persisted, or used for monitoring.

## Choose a logger

[`Trainer`](../api/trainer.md#phijax.training.Trainer) with `logger=True` uses TensorBoard when it is installed and CSV otherwise. Local runs use
`<default_root_dir>/phijax_logs/version_N/`.

```python
from phijax import Trainer

trainer = Trainer(
    max_steps=10_000,
    logger=True,
    default_root_dir="outputs",
)
```

Open TensorBoard logs with:

```bash
tensorboard --logdir outputs/phijax_logs
```

Pass `logger=False` or `logger=None` to disable experiment logging. Pass one logger or an iterable to select explicit
backends. Logger construction is lazy; files and remote runs are created when the Trainer task starts on global rank
zero.

## Record run settings

Pass resolved settings through `hyperparameters` so the logger stores them before the first update.

```python
result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=0,
    hyperparameters={"model": {"width": 128}, "seed": 0},
)
```

Hydra projects can convert a composed config with
[`to_hyperparameters()`](../api/configuration.md#phijax.integrations.hydra.to_hyperparameters).

## Select callbacks and displays

The Trainer adds TQDM progress and a plain model summary by default. Supply Rich replacements explicitly, or disable
either display through the Trainer.

```python
from phijax import Trainer
from phijax.callbacks import RichModelSummary, RichProgressBar

callbacks = (
    RichModelSummary(),
    RichProgressBar(total=10_000, refresh_rate=10),
)
trainer = Trainer(
    max_steps=10_000,
    callbacks=callbacks,
    enable_progress_bar=True,
    enable_model_summary=True,
)
```

Set `enable_progress_bar=False` or `enable_model_summary=False` for quiet jobs. Supplying multiple callbacks of the
same display type raises an error.

## Change metric destinations

Every scalar returned by the compiled step is sent to configured loggers. Total loss, individual losses, and balancer
weights also appear in the progress bar. Override these defaults from `on_train_batch_end()`:

```python
def on_train_batch_end(self, model_state, context):
    model_state, metrics = super().on_train_batch_end(model_state, context)
    self.log("train/loss/pde/heat", metrics["train/loss/pde/heat"], prog_bar=False)
    self.log("train/residual/mean_abs", metrics["train/residual/mean_abs"], prog_bar=True)
    return model_state, metrics
```

[`BasePhiModule.log()`](../api/module.md#phijax.core.BasePhiModule.log) queues the device value. It does not write to a logger or
transfer the value immediately. It is valid only during the module batch-end hook. Array diagnostics stay in
`callback_metrics` and must be reduced to a scalar before they can be logged or displayed.

The Trainer exposes three read-only metric mappings:

| Property               | Contents                                      |
| ---------------------- | --------------------------------------------- |
| `callback_metrics`     | Every scalar and array available to callbacks |
| `logged_metrics`       | Scalars selected for experiment loggers       |
| `progress_bar_metrics` | Scalars selected for the terminal display     |

## Monitor the learning rate

[`LearningRateMonitor`](../api/callbacks.md#phijax.callbacks.LearningRateMonitor) reports the rate used by each completed Optax update. It
requires a configured logger.

```python
from phijax import Trainer
from phijax.callbacks import LearningRateMonitor

lr_monitor = LearningRateMonitor(
    schedule,
    optimizer_name="Adam",
    logging_interval=None,
)
trainer = Trainer(max_steps=10_000, callbacks=(lr_monitor,), logger=True)
```

The default metric name is `optimizer/lr-Adam`. `None` follows `trainer.log_every_n_steps`, `"step"` records every
update, and `"epoch"` records only the end of the fit call.

## Retain the best checkpoints

Use [`ModelCheckpoint`](../api/callbacks.md#phijax.callbacks.ModelCheckpoint) with a scalar metric produced by the module or a callback:

```python
from phijax import Trainer
from phijax.callbacks import ModelCheckpoint

checkpoint = ModelCheckpoint(
    checkpoint_io,
    monitor="train/loss",
    mode="min",
    save_top_k=3,
    save_last=True,
)
trainer = Trainer(max_steps=10_000, callbacks=(checkpoint,))
```

The callback exposes `best_model_path`, `best_model_score`, and `last_model_path`. A missing or non-scalar monitored
metric raises instead of silently disabling ranking.

## Avoid unnecessary synchronization

Logging follows `log_every_n_steps`, and progress follows its callback refresh rate. Keep arrays on device between
those events. Converting a device value to Python or NumPy from a per-step callback intentionally waits for pending
device work and can reduce throughput.

## Next steps

- [Logger API](../api/loggers.md)
- [Callback API](../api/callbacks.md)
- [Checkpointing](../api/checkpointing.md)
- [Trainer hooks](../api/hooks.md)
