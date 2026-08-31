# PhiModule

`BasePhiModule` connects application computations to the Trainer. It defines model application, unweighted losses,
prediction behavior, metric formatting, and application hooks. It does not own an optimizer or loss balancer.

The public contracts live in `phijax.core`. Their host lifecycle defaults are separated internally from numerical
module behavior, but applications continue to customize both by subclassing `BasePhiModule`.

## Lifecycle ordering

Callbacks run before the corresponding module hook:

```text
callbacks.on_fit_start           -> module.on_fit_start
callbacks.on_train_batch_start   -> module.on_train_batch_start
compiled train step
callbacks.on_train_batch_end     -> module.on_train_batch_end
callbacks.on_fit_end             -> module.on_fit_end

callbacks.on_predict_start       -> module.on_predict_start
callbacks.on_predict_epoch_start -> module.on_predict_epoch_start
callbacks.on_predict_batch_start -> module.on_predict_batch_start
module.predict_step
callbacks.on_predict_batch_end   -> module.on_predict_batch_end
callbacks.on_predict_epoch_end   -> module.on_predict_epoch_end
callbacks.on_predict_end         -> module.on_predict_end
```

Module hooks run on the Python host. Training hooks return explicit replacement values. These values must keep the
PyTree structure and fixed shapes expected by compiled functions. Prediction lifecycle hooks only observe state and
return `None`. Override `predict_step` to transform each output batch before collection and artifact writing.

## Logging from a module

The compiled `training_step()` returns numerical losses and diagnostics. The built-in `PhiModule.on_train_batch_end()`
uses `self.log()` to send every scalar to experiment loggers. This includes individual `train/loss/<name>` losses and
`train/weight/<name>` balancer weights. The total loss, individual losses, and balancer weights also appear in the
progress bar by default. Override their destinations after calling the standard hook:

```python
def on_train_batch_end(self, model_state, context):
    model_state, metrics = super().on_train_batch_end(model_state, context)
    self.log("train/loss/pde/heat", metrics["train/loss/pde/heat"], prog_bar=False)
    self.log("train/weight/pde/heat", metrics["train/weight/pde/heat"], prog_bar=False)
    self.log("train/residual/mean_abs", metrics["train/residual/mean_abs"], prog_bar=True)
    return model_state, metrics
```

`self.log()` does not write immediately. It records device values for the Trainer, which converts them only at the
configured logging or progress-refresh cadence. It is intentionally unavailable inside JAX transformations.

## Module blueprints

The standard `PhiModule` is an uninitialized blueprint containing a `ModelFactory` and objective:

```python
module = PhiModule(model_factory, objective, name="heat")
result = trainer.fit(module, datamodule=data_module, optimizer=optimizer, seed=0)
```

`Trainer.fit()` supplies model initialization values and returns a separately bound module as `result.module`. The
blueprint is not mutated. Its objective provides ordered `loss_names` and `batch_keys`, allowing the Trainer to build
the balancer, training plan, and DataModule source.

## Implementing a custom module

Implement `loss_names`, `forward`, and `training_step`. Implement `residual_stream` when using derivative-based loss
balancing. Fully custom bound modules use `Trainer.fit_state()` with an explicit `TrainingPlan` and `TrainState`.

An application can construct any `BasePhiModule` subclass. Keep module selection separate from optimizer and
loss-balancer selection. See [Configuration integrations](configuration.md) when assembling modules with Hydra.

```python
class HeatModule(BasePhiModule):
    @property
    def loss_names(self) -> tuple[str, ...]:
        return ("initial/u", "boundary/u", "pde/heat")

    def forward(self, model_state, inputs):
        return self.model_apply(model_state, inputs)

    def training_step(self, model_state, batches):
        return self.objective.losses(self.model_apply, model_state, batches)
```

## Public API

::: phijax.core.PhiModuleContext

::: phijax.core.BasePhiModule

options:
members:
\- loss_names
\- batch_keys
\- forward
\- training_step
\- log
\- format_training_metrics
\- residual_stream
\- summarize_model
\- predict_step
\- on_predict_start
\- on_predict_epoch_start
\- on_predict_batch_start
\- on_predict_batch_end
\- on_predict_epoch_end
\- on_predict_end
\- setup
\- on_fit_start
\- on_train_batch_start
\- on_train_batch_end
\- on_fit_end
\- on_exception
\- teardown

::: phijax.core.PhiModule

options:
members:
\- loss_names
\- batch_keys
\- prepare_model
\- forward
\- summarize_model
\- training_step
\- residual_stream
