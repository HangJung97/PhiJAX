# Configuration integrations

PhiJAX does not require a particular configuration layout. `phijax.integrations.hydra` can build typed runtime objects
from project-owned Hydra configs. `phijax.integrations.omegaconf` provides reusable resolvers.

This page documents the integration functions provided by the package. For a complete config tree, CLI entrypoints,
and experiment composition, use the
[PhiJAX Hydra template](https://github.com/HangJung97/phijax-hydra-template).

## Hydra factories

The factory helpers instantiate callbacks, loggers, a Trainer, DataModule, lazy model factory, objective, PhiModule,
loss balancer, and optimizer. A balancer config is directly instantiable and does not need an intermediate `factory`
field:

```yaml
model:
  balancer:
    _target_: phijax.balancers.ExactNTKBalancer
    update_every_n_steps: 100
    update_start_step: 100
    kernel_size: 256
    kernel_chunk_size: 1
```

The project passes `model.balancer` to `instantiate_balancer()`. Use `Trainer.fit()` for ordinary training or the core
`build_training_plan()` function when an advanced `fit_state()` workflow needs an explicit plan.

::: phijax.integrations.hydra.instantiate_data_module

::: phijax.integrations.hydra.instantiate_model_factory

::: phijax.integrations.hydra.instantiate_module

::: phijax.integrations.hydra.instantiate_objective

::: phijax.integrations.hydra.instantiate_balancer

::: phijax.integrations.hydra.instantiate_optimizer

Callback and logger entries are active when they are present in the composed configuration. To disable one, remove its
entry from the config or use Hydra's deletion override:

```bash
python train.py '~callbacks.early_stopping'
```

The complete `callbacks` or `logger` group may be `null`. Named entries must define `_target_`; do not add an `enabled`
field or leave a named entry set to `null`.

::: phijax.integrations.hydra.instantiate_callbacks

::: phijax.integrations.hydra.instantiate_loggers

::: phijax.integrations.hydra.instantiate_trainer

::: phijax.integrations.hydra.build_trainer

Projects that need the individual construction steps can inject the already constructed services directly:

```python
from phijax.integrations.hydra import (
    instantiate_callbacks,
    instantiate_loggers,
    instantiate_trainer,
    to_hyperparameters,
)

callbacks = instantiate_callbacks(config.get("callbacks"))
loggers = instantiate_loggers(config.get("logger"))
trainer = instantiate_trainer(config.trainer, callbacks, logger=loggers)

hyperparameters = to_hyperparameters(config)
result = trainer.fit(module, datamodule=data, optimizer=optimizer, seed=seed, hyperparameters=hyperparameters)
```

Logger constructors do not acquire local or remote resources. The Trainer starts and closes them only on global rank
zero when the task runs.

::: phijax.integrations.hydra.to_hyperparameters

## OmegaConf resolvers

Project entrypoints should register resolvers before Hydra composition:

```python
from phijax.integrations.omegaconf import register_omegaconf_resolvers

register_omegaconf_resolvers()
```

| Resolver     | Example                         | Purpose                     |
| ------------ | ------------------------------- | --------------------------- |
| `math`       | `${math:pi}`                    | Public Python `math` member |
| `op`         | `${op:truediv,0.01,${math:pi}}` | Public `operator` function  |
| `op.ternary` | `${op.ternary,true,a,b}`        | Conditional value           |
| `tuple`      | `${tuple:1,2,3}`                | Tuple rather than list      |
| `call`       | `${call:module.function,arg}`   | Trusted imported callable   |
| `assert`     | `${assert:true}`                | Strict or warning assertion |

Configuration can execute imported callables through Hydra targets and the `call` resolver. Compose only trusted
configuration files.

::: phijax.integrations.omegaconf.register_omegaconf_resolvers

::: phijax.integrations.omegaconf.import_from_module
