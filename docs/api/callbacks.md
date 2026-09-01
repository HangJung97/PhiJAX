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
  module logging collection
  on_train_metrics
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
`on_train_metrics` then receives the complete metric mapping. It is the appropriate hook for checkpoint ranking and
display updates that need metrics from every provider.

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

`mode` accepts `"min"` or `"max"`, represented by the public `MonitorMode` alias. Use `"min"` for losses and
`"max"` for scores where larger values are better.

::: phijax.callbacks.EarlyStopping

## Learning-rate monitoring

`LoggingInterval` represents the explicit `"step"` and `"epoch"` interval values. Passing `None` follows the Trainer
logging cadence.

`LearningRateMonitor` evaluates the Optax schedule at the optimizer count for each completed update. It sends
`optimizer/lr-Adam` to the configured loggers when `optimizer_name="Adam"`. `logging_interval="step"` evaluates every
step, while
`logging_interval="epoch"` evaluates once at the end of fitting. The default `None` follows
`trainer.log_every_n_steps` and always records the final rate. Unlike a Lightning scheduler, a raw Optax schedule has
no `interval` field. PhiJAX therefore avoids evaluating the schedule on steps that will not be logged.

Like Lightning, this callback requires an experiment logger. Fitting raises before requesting the first batch when
`LearningRateMonitor` is enabled with `logger=False`, `logger=None`, or an empty logger collection. The default
`logger=True` is valid.

Like Lightning, optimizer names identify learning-rate series, while `log_momentum` and `log_weight_decay` add
`-momentum` and `-weight_decay` suffixes. PhiJAX requires `optimizer_name` because Optax transformations do not retain
their factory name. Metrics use the `optimizer/` group by default; set `log_key_prefix` to replace it or pass `None`
to disable grouping. Optional momentum and weight-decay values must be supplied because Optax transformations do not
expose inspectable parameter groups.

::: phijax.callbacks.LearningRateMonitor

## Model checkpoints

`ModelCheckpoint` delegates storage to `CheckpointIO`; the default backend uses Orbax. A Trainer accepts only one
checkpoint callback. The callback opens its backend when fitting starts and closes it during teardown. Custom backends
must support repeated `open()` and `close()` calls and must be able to reopen for later tasks.

Set `monitor`, `mode`, and `save_top_k` to retain the best checkpoints by an available scalar metric. `save_last=True`
keeps the terminal state independently. Public `best_model_path`, `best_model_score`, and `last_model_path` attributes
expose the resolved results. Checkpoints also persist callback state, including ranking and learning-rate bookkeeping.

::: phijax.callbacks.CheckpointIO

::: phijax.callbacks.ModelCheckpoint

## Model summary

`Trainer(enable_model_summary=True)` adds `ModelSummary` when no summary callback is supplied. Pass
`RichModelSummary` to replace it, or set `enable_model_summary=False` for no automatic summary. A Trainer rejects
multiple summary callbacks and prints summaries only on global rank zero.

::: phijax.callbacks.ModelSummary

::: phijax.callbacks.RichModelSummary

## Progress bar

When no progress callback is supplied, `Trainer(enable_progress_bar=True)` adds `TQDMProgressBar`. Set
`enable_progress_bar=False` for quiet library, test, or batch-scheduler execution.

All progress callbacks also provide `enable()`, `disable()`, and `is_enabled` for temporary runtime control. Most
applications should use the Trainer option instead of managing callback state directly.

::: phijax.callbacks.ProgressBar

::: phijax.callbacks.TQDMProgressBar

Supplying `RichProgressBar`, which extends `ProgressBar`, replaces TQDM. A Trainer accepts at most one progress
callback, avoiding duplicate output. Both implementations refresh device metrics only at their configured interval.

By default, both displays show `train/loss`, every `train/loss/<name>`, every `train/weight/<name>`, and the primary
logger version as `v_num`. Learning rates and other diagnostics still reach loggers without entering the display.
Override `ProgressBar.get_metrics()` to change standard fields, or pass `metric_names` for an exact ordered selection.
During prediction, the callbacks use `PredictionContext.total_batches` without transferring prediction values to the
host.

## Module logging and diagnostics

Every scalar from the compiled step is logged by default. Total loss, individual losses, and balancer weights also
appear in the progress bar. `self.log()` can change these destinations from the module batch-end hook without moving
logging into JIT-compiled code.

The Trainer exposes the latest complete, persisted, and displayed mappings as `callback_metrics`, `logged_metrics`,
and `progress_bar_metrics`. Callback `training_metrics()` values are logger-only by default. See
[Log and monitor training](../guides/logging.md) for examples.

::: phijax.callbacks.RichProgressBarTheme

::: phijax.callbacks.RichProgressBar

## Compose callbacks

Pass callbacks to the Trainer in dispatch order. A Trainer accepts at most one progress callback, one model-summary
callback, one checkpoint callback, and one prediction writer. See [Log and monitor training](../guides/logging.md) for
composition examples and [Configuration integrations](configuration.md) for optional Hydra construction.
