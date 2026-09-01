# PhiModule

`PhiModule` connects a model factory to an objective. It describes the computation that the
[`Trainer`](trainer.md#phijax.training.Trainer) initializes and runs. Optimizer, loss-balancer, and PRNG state remain
outside the module in [`TrainState`](training.md#phijax.training.TrainState).

Use `PhiModule` for the standard objective-backed workflow. Subclass `BasePhiModule` only when an application needs a
different numerical contract.

## Basic use

Create an uninitialized module blueprint, fit it, and use the bound module returned in `FitResult` for prediction:

```python
module = PhiModule(model_factory, objective, name="Heat PINN")

result = trainer.fit(
    module,
    datamodule=data_module,
    optimizer=optimizer,
    seed=0,
)
predictions = trainer.predict(result, datamodule=data_module)
```

`Trainer.fit()` supplies the model key, precision policy, and optional input statistics. It does not modify the
original blueprint. See the [heat-equation quickstart](../getting-started/quickstart.md) for a complete runnable
example.

## Responsibilities

| Component     | Responsibility                                                        |
| ------------- | --------------------------------------------------------------------- |
| `PhiModule`   | Model application, objective losses, prediction, and metric routing   |
| Model factory | Model initialization and explicit model state                         |
| Objective     | Named unweighted losses and required DataModule batch keys            |
| `Trainer`     | Runtime assembly, hooks, device placement, logging, and checkpointing |
| `TrainState`  | Model, optimizer, balancer, step, precision, and explicit PRNG arrays |
| Loss balancer | Combining named losses and scheduling adaptive weight updates         |

This separation keeps the compiled computation functional. A module never owns mutable optimizer or balancer state.

## Module blueprints

The standard `PhiModule` accepts either a lazy [`ModelFactory`](models/index.md#phijax.models.ModelFactory) or an
initialized [`InitializedModel`](models/index.md#phijax.models.InitializedModel), together with an
[`Objective`](objectives.md#phijax.objectives.Objective).

The objective supplies ordered `loss_names` and `batch_keys`. The Trainer uses them to create default static loss
weights and request matching batches from the DataModule. During fitting, `prepare_model()` creates a shallow bound
copy containing the pure model application and model summary function. The resulting bound module is available as
`result.module`.

## Logging metrics

The compiled `training_step()` returns numerical losses and diagnostics. The standard `PhiModule.on_train_batch_end()`
uses `self.log()` to route every scalar to configured experiment loggers. By default, these values also appear in the
progress bar:

- `train/loss`;
- each `train/loss/<name>` objective loss;
- each `train/weight/<name>` balancer weight.

Override a destination after calling the standard hook:

```python
def on_train_batch_end(self, model_state, context):
    model_state, metrics = super().on_train_batch_end(model_state, context)
    self.log("train/loss/pde/heat", metrics["train/loss/pde/heat"], prog_bar=False)
    self.log("train/weight/pde/heat", metrics["train/weight/pde/heat"], prog_bar=False)
    self.log("train/residual/mean_abs", metrics["train/residual/mean_abs"], prog_bar=True)
    return model_state, metrics
```

`self.log()` records the device value without writing immediately. The Trainer converts it only at the configured
logging or progress-refresh cadence. The method is available during `on_train_batch_end()` and must not be called from
inside a JAX transformation. See [Logging and monitoring](../guides/logging.md) for logger and progress-bar behavior.

## Custom modules

Subclass `BasePhiModule` when the standard model-factory and objective composition is not sufficient. Implement
`loss_names`, `forward()`, and `training_step()`. Implement `residual_stream()` when an adaptive loss balancer needs
raw residuals or output streams.

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

A fully bound custom module uses [`Trainer.fit_state()`](trainer.md#phijax.training.Trainer.fit_state) with an explicit
`TrainingPlan` and `TrainState`. Keep optimizer and loss-balancer selection outside the module.

## Lifecycle and hooks

Callbacks run in declaration order before the matching module hook. The main fit ordering is:

```text
callbacks.setup                 -> module.setup
callbacks.on_fit_start          -> module.on_fit_start
callbacks.on_train_batch_start  -> module.on_train_batch_start
compiled training step
callbacks.on_train_batch_end    -> module.on_train_batch_end
callbacks.on_fit_end            -> module.on_fit_end
callbacks.teardown              -> module.teardown
```

Module hooks run on the Python host. Training hooks return explicit replacement values and must preserve the PyTree
structure and fixed shapes expected by compiled functions. Prediction hooks observe state; override `predict_step()`
to transform output batches before collection or artifact writing.

See [Trainer and module hooks](hooks.md) for the complete fit, prediction, interruption, and exception lifecycles.

## API reference

### Standard module

::: phijax.core.PhiModule
    options:
      members:
        - loss_names
        - batch_keys
        - prepare_model
        - summarize_model
        - on_train_batch_end

### Custom module contract

::: phijax.core.BasePhiModule
    options:
      members:
        - loss_names
        - batch_keys
        - forward
        - training_step
        - log
        - format_training_metrics
        - residual_stream
        - summarize_model
        - predict_step
        - setup
        - on_fit_start
        - on_train_batch_start
        - on_train_batch_end
        - on_fit_end
        - on_predict_start
        - on_predict_epoch_start
        - on_predict_batch_start
        - on_predict_batch_end
        - on_predict_epoch_end
        - on_predict_end
        - on_exception
        - teardown

### Hook context

::: phijax.core.PhiModuleContext
