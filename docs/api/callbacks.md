# Callbacks

Callbacks implement independent host-side monitoring and side effects. They do not belong in transformed JAX
functions, and numerical model changes should use explicit module hook return values instead.

## Contexts and hook order

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

on_postprocessing_start
on_postprocessing_end
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

The default prediction-only callback suite contains exactly one `PredictionWriter`. It writes the canonical NPZ after
the final host outputs have been concatenated and can optionally add a MATLAB sidecar. The fitting callback suite also
contains this writer but enables it only when `predict=true`, allowing the same Trainer and in-memory state to continue
into prediction after fitting completes or is stopped gracefully with `Ctrl+C`. `SIGTERM` skips prediction.

::: phijax.callbacks.PredictionWriter

## Early stopping

::: phijax.callbacks.EarlyStopping

## Learning-rate monitoring

`LearningRateMonitor` evaluates the configured Optax schedule at the optimizer count used by each completed update.
It publishes `train/lr` through the ordinary logger stream. `logging_interval="step"` evaluates every optimizer step,
while `logging_interval="epoch"` evaluates once at the end of the fit pass. The default `None` follows
`trainer.log_every_n_steps` and always records the terminal rate. This differs slightly from Lightning because a raw
Optax schedule has no scheduler object carrying an individual `interval` field, and it avoids dispatching a separate
JAX schedule operation on unlogged steps.

Like Lightning, `log_momentum` and `log_weight_decay` enable `-momentum` and `-weight_decay` metrics, and
`log_key_prefix` is prepended verbatim to every key. PhiJAX supplies the optional values explicitly because an Optax
transformation has no inspectable parameter groups.

::: phijax.callbacks.LearningRateMonitor

## Model checkpoints

`ModelCheckpoint` delegates persistence to `CheckpointIO`; the default implementation uses Orbax. Only one checkpoint
callback may be attached to a trainer.

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

## Project composition

Projects may assign each callback an individual Hydra config. A defaults list can compose fitting callbacks, while
`predict.yaml` composes prediction-only callbacks. `instantiate_enabled` ignores entries without `_target_` and entries
with `enabled: false`.

```yaml
callbacks:
  early_stopping:
    enabled: true
    monitor: train/loss
    patience: 1000
```
