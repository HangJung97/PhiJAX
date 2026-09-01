# Trainer and module hooks

PhiJAX keeps numerical updates inside JAX transformations and runs lifecycle hooks on the Python host. Use hooks for
orchestration, monitoring, explicit batch or model-state replacement, and external side effects. Hooks must not mutate
traced arrays or change a compiled PyTree's structure between steps.

For every matching event, callbacks run in declaration order before the `BasePhiModule` hook. This follows Lightning's
default ordering and gives application code a predictable final extension point.

`fit()` first prepares the DataModule, initializes and binds the model, and resolves state and a training plan.
`fit_state()` accepts those resolved objects directly. Both then dispatch the same lifecycle below.

## Fit lifecycle

```text
Trainer.fit / Trainer.fit_state
|
+-- Callback.setup()                         declaration order
+-- BasePhiModule.setup()
+-- logger.log_hyperparams(...)
+-- Callback.on_fit_start(context)
|   +-- ModelCheckpoint opens its backend
+-- BasePhiModule.on_fit_start(state, context) -> model_state
|
+-- repeat for each training batch
|   +-- DataModule batch source
|   +-- precision cast and device placement
|   +-- Callback.on_train_batch_start(context)
|   +-- BasePhiModule.on_train_batch_start(...) -> model_state, batch
|   +-- compiled training step
|   |   +-- unweighted module losses
|   |   +-- loss-balancer update and weighted total
|   |   +-- parameter gradients
|   |   +-- Optax update
|   +-- Callback.on_train_batch_end(context) -> stop request
|   +-- BasePhiModule.on_train_batch_end(...) -> model_state, metrics
|   +-- collect module self.log(...) declarations
|   +-- Callback.training_metrics(context) -> additional metrics
|   +-- Callback.on_train_metrics(context)
|   +-- logging at the configured cadence
|
+-- Callback.on_fit_end(context)
+-- BasePhiModule.on_fit_end(state, context) -> model_state
+-- logger.finalize("success")
+-- Callback.teardown()
|   +-- ModelCheckpoint waits for writes and closes its backend
+-- BasePhiModule.teardown()
```

The module's `on_fit_start`, `on_train_batch_start`, `on_train_batch_end`, and `on_fit_end` hooks return explicit
replacement values. This is how application code changes numerical state without hidden mutation.
`on_train_batch_start` may also replace the placed batch. Its PyTree structure and array shapes must still match the
compiled step.

`TrainerContext.batch` contains the device-placed batch during callback `on_train_batch_start()` and is `None` for
other fit hooks. This supports occasional debugging or inspection without making batch data part of every callback
event. Copy arrays to the host only when needed because `jax.device_get()` synchronizes accelerator execution.

Callback `on_train_batch_end()` returns `True` to request a clean stop. The remaining callbacks and the module hook
still finish for that step. During the module hook, `self.log()` may change logger and progress destinations or add a
host-computed scalar. `training_metrics()` then adds uniquely named logger-only callback metrics. It cannot replace an
existing metric. `on_train_metrics()` receives the complete mapping; use it for displays, monitored checkpoints, or
diagnostics that need every provider's values.

`predict()` obtains its bound module and state from `FitResult`. `predict_state()` accepts them explicitly. Both use the
same prediction lifecycle.

## Prediction lifecycle

```text
Trainer.predict / Trainer.predict_state
|
+-- optional DataModule.prepare_stage("predict")
+-- optional checkpoint weight restoration
+-- Callback.setup()
+-- BasePhiModule.setup()
+-- Callback.on_predict_start(context)
+-- BasePhiModule.on_predict_start(state, context)
+-- Callback.on_predict_epoch_start(context)
+-- BasePhiModule.on_predict_epoch_start(state, context)
|
+-- repeat for each prediction batch
|   +-- precision cast and device placement
|   +-- Callback.on_predict_batch_start(context)
|   +-- BasePhiModule.on_predict_batch_start(state, context)
|   +-- BasePhiModule.predict_step(state, batch)
|   +-- remove padded values selected by `mask`
|   +-- Callback.on_predict_batch_end(context)
|   +-- BasePhiModule.on_predict_batch_end(state, context)
|   +-- optional device-to-host collection
|
+-- concatenate collected outputs
+-- Callback.on_predict_epoch_end(context)
+-- BasePhiModule.on_predict_epoch_end(state, context)
+-- Callback.on_predict_end(context)
+-- BasePhiModule.on_predict_end(state, context)
+-- logger.finalize("success")
+-- Callback.teardown()
+-- BasePhiModule.teardown()
+-- optional DataModule.teardown_stage("predict")
```

Prediction hooks only observe state. Override `predict_step()` to transform model outputs. Batch-end hooks receive
valid, unpadded outputs. Epoch-end and predict-end hooks receive the joined host array when
`return_predictions=True`. If a DataModule has no prediction source, `Trainer.predict()` returns `None` and skips the
prediction hooks.

## Exceptions and interruption

If a task fails after setup, PhiJAX calls each initialized callback's `on_exception()` before
`BasePhiModule.on_exception()`. It finalizes the logger with `"failed"` and then runs teardown. A resource is only torn
down if its setup completed.

During fitting, the first `Ctrl+C` preserves the last completed state and returns `FitResult(interrupted=True)`.
Exception hooks, logger finalization, and teardown still run. `SIGTERM` performs the same cleanup but exits instead of
returning. A repeated signal exits immediately.

## Choosing an extension point

| Requirement                                  | Extension point                               |
| -------------------------------------------- | --------------------------------------------- |
| Change forward prediction                    | `BasePhiModule.forward()` or `predict_step()` |
| Define unweighted objective losses           | `BasePhiModule.training_step()`               |
| Replace model state before fitting           | module `on_fit_start()`                       |
| Change a placed training batch               | module `on_train_batch_start()`               |
| Route or add scalar training metrics         | module `self.log()` in `on_train_batch_end()` |
| Observe pre-routing step metrics             | callback `on_train_batch_end()`               |
| Add host-computed scalar metrics             | callback `training_metrics()`                 |
| Observe the complete metric mapping          | callback `on_train_metrics()`                 |
| Stop cleanly from a monitoring policy        | callback `on_train_batch_end()`               |
| Transform prediction outputs                 | `BasePhiModule.predict_step()`                |
| Stream or save prediction outputs            | callback `on_predict_batch_end()` or `_end()` |
| Release application or integration resources | `teardown()`                                  |

`on_postprocessing_start()` and `on_postprocessing_end()` remain callback contracts for project-owned orchestration;
the core `Trainer.fit()` and `Trainer.predict()` methods do not dispatch them.

See the [PhiModule API](module.md), [callbacks API](callbacks/index.md), and [Trainer API](trainer.md) for signatures and
context fields.
