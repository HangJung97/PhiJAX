# Callbacks

Callbacks handle host-side monitoring and side effects. Do not call them inside transformed JAX functions. Use module
hook return values for numerical model changes.

## Contexts and hook order

The complete dispatch sequence and extension-point guidance are documented in
[Trainer and module hooks](hooks.md).

```text
setup
on_fit_start
  on_train_batch_start
  compiled update
  on_train_batch_end -> bool
  training_metrics   -> scalar metric mapping
on_fit_end

on_predict_start
  on_predict_epoch_start
    on_predict_batch_start
    on_predict_batch_end
  on_predict_epoch_end
on_predict_end

on_postprocessing_start  project-owned orchestration only
on_postprocessing_end    project-owned orchestration only
on_exception
teardown
```

`on_train_batch_end` returns `True` to request a clean early stop. Callbacks run before corresponding `BasePhiModule`
hooks.

`training_metrics` is a read-only extension point evaluated after the module batch-end hook. Metric names must be
non-empty and may not replace metrics produced by the compiled step, the module, or another callback.

::: phijax.callbacks.TrainerContext

::: phijax.callbacks.PredictionContext

::: phijax.callbacks.PostprocessingContext

::: phijax.callbacks.Callback

`PredictionContext.pool` exposes the immutable pool carried by a `PredictionBatchSource`, while `is_global_zero`
allows artifact callbacks to avoid duplicate distributed writes.

## Prediction writer

The default prediction callback suite contains one `PredictionWriter`. It writes the canonical NPZ after joining all
host outputs and can also write a MATLAB file. The fit callback suite enables this writer only when `predict=true`.
This lets prediction reuse the same Trainer and in-memory state after training finishes or stops with `Ctrl+C`.
`SIGTERM` skips prediction.

::: phijax.callbacks.PredictionWriter

## Early stopping

::: phijax.callbacks.EarlyStopping

## Learning-rate monitoring

`LearningRateMonitor` evaluates the Optax schedule at the optimizer count for each completed update. It sends
`train/lr` to the configured loggers. `logging_interval="step"` evaluates every step, while
`logging_interval="epoch"` evaluates once at the end of fitting. The default `None` follows
`trainer.log_every_n_steps` and always records the final rate. Unlike a Lightning scheduler, a raw Optax schedule has
no `interval` field. PhiJAX therefore avoids evaluating the schedule on steps that will not be logged.

Like Lightning, `log_momentum` and `log_weight_decay` add `-momentum` and `-weight_decay` metrics.
`log_key_prefix` is added to every key. These optional values must be supplied because Optax transformations do not
expose inspectable parameter groups.

::: phijax.callbacks.LearningRateMonitor

## Model checkpoints

`ModelCheckpoint` delegates storage to `CheckpointIO`; the default backend uses Orbax. A Trainer accepts only one
checkpoint callback. The callback opens its backend when fitting starts and closes it during teardown. Custom backends
must support repeated `open()` and `close()` calls and must be able to reopen for later tasks.

::: phijax.callbacks.CheckpointIO

::: phijax.callbacks.ModelCheckpoint

## Model summary

::: phijax.callbacks.RichModelSummary

## Progress bar

When `metric_names` is omitted, the Rich progress bar automatically discovers the `train/loss` and `train/weight`
namespaces. During prediction it uses `PredictionContext.total_batches` to display finite batch progress without
transferring prediction values to the host. It should usually run only on global rank zero.

::: phijax.callbacks.RichProgressBarTheme

::: phijax.callbacks.RichProgressBar

## Compose callbacks

Pass callbacks to the Trainer in dispatch order:

```python
from phijax.callbacks import EarlyStopping, RichProgressBar
from phijax.training import Trainer

callbacks = (
    EarlyStopping(monitor="train/loss", patience=1_000),
    RichProgressBar(),
)
trainer = Trainer(max_steps=10_000, callbacks=callbacks)
```

See [Configuration integrations](configuration.md) for optional Hydra-based callback construction.
