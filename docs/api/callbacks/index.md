# Callbacks

Callbacks handle host-side monitoring and side effects. Do not call them inside transformed JAX functions. Use module
hook return values for numerical model changes.

## Callback overview

| Task                           | Reference                               |
| ------------------------------ | --------------------------------------- |
| Stop or monitor optimization   | [Monitoring callbacks](monitoring.md)   |
| Change terminal displays       | [Display callbacks](display.md)         |
| Save prediction artifacts      | [Prediction callbacks](prediction.md)   |
| Save or rank training state    | [Checkpointing](../checkpointing.md)    |
| Implement a new lifecycle hook | [Callback contract](#callback-contract) |

A Trainer accepts at most one progress callback, one model-summary callback, one checkpoint callback, and one
prediction writer.

## Contexts and hook order

The complete dispatch sequence and extension-point guidance are documented in
[Trainer and module hooks](../hooks.md).

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

`on_train_batch_end()` returns `True` to request a clean early stop. Callbacks run before corresponding
`BasePhiModule` hooks.

`training_metrics()` is a read-only extension point evaluated after the module batch-end hook. Metric names must be
non-empty and may not replace metrics produced by the compiled step, the module, or another callback.
`on_train_metrics()` then receives the complete metric mapping. Use it for checkpoint ranking and display updates that
need values from every provider.

`PredictionContext.pool` exposes the immutable pool carried by a `PredictionBatchSource`. `is_global_zero` allows
artifact callbacks to avoid duplicate distributed writes.

## Module logging and diagnostics

Every scalar from the compiled step is logged by default. Total loss, individual losses, and balancer weights also
appear in the progress bar. `self.log()` can change these destinations from the module batch-end hook without moving
logging into JIT-compiled code.

The Trainer exposes the latest complete, persisted, and displayed mappings as `callback_metrics`, `logged_metrics`,
and `progress_bar_metrics`. Callback `training_metrics()` values are logger-only by default. See
[Logging and monitoring](../../guides/logging.md) for examples.

## Compose callbacks

Pass callbacks to the Trainer in dispatch order:

```python
trainer = Trainer(
    max_steps=10_000,
    callbacks=(early_stopping, learning_rate_monitor, progress_bar),
)
```

See [Configuration integrations](../configuration.md) for optional Hydra construction.

## Callback contract

::: phijax.callbacks.TrainerContext

::: phijax.callbacks.PredictionContext

::: phijax.callbacks.PostprocessingContext

::: phijax.callbacks.Callback
