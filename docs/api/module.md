# PhiModule

`BasePhiModule` connects application computations to the Trainer. It defines model application, unweighted losses,
prediction behavior, metric formatting, and application hooks. It does not own an optimizer or loss balancer.

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

## Implementing a custom module

Implement `loss_names`, `forward`, and `training_step`. Implement `residual_stream` when using derivative-based loss
balancing. The standard `PhiModule` already delegates these operations to a model application callable and objective.

Hydra-based projects can select the standard implementation through their own module config group:

```yaml
defaults:
  - module: phi_module
```

An application can point `_target_` to any `BasePhiModule` subclass. Its entrypoints inject `model_apply`, `objective`,
`name`, and `model_summary`, so a custom constructor should accept those arguments. Module selection remains separate
from optimizer and loss-balancer selection.

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

::: phijax.module.PhiModuleContext

::: phijax.module.BasePhiModule
options:
members:
\- loss_names
\- forward
\- training_step
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

::: phijax.module.PhiModule
options:
members:
\- loss_names
\- forward
\- summarize_model
\- training_step
\- residual_stream
