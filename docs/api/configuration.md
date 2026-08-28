# Configuration integrations

PhiJAX does not require a particular configuration layout. `phijax.integrations.hydra` can build typed runtime objects
from project-owned Hydra configs. `phijax.integrations.omegaconf` provides reusable resolvers.

This page documents the integration functions provided by the package. For a complete config tree, CLI entrypoints,
and experiment composition, use the
[PhiJAX Hydra template](https://github.com/HangJung97/phijax-hydra-template).

## Hydra factories

The factory helpers instantiate callbacks, loggers, a Trainer, DataModule, initialized model, objective, PhiModule,
loss balancer, and optimizer. `configure_training()` combines those objects into a `TrainingPlan`. The Trainer then
constructs and places the DataModule source during `fit()`.

::: phijax.integrations.hydra.instantiate_data_module

::: phijax.integrations.hydra.instantiate_model

::: phijax.integrations.hydra.instantiate_module

::: phijax.integrations.hydra.instantiate_objective

::: phijax.integrations.hydra.instantiate_balancer

::: phijax.integrations.hydra.instantiate_optimizer

::: phijax.integrations.hydra.instantiate_callbacks

::: phijax.integrations.hydra.instantiate_enabled

::: phijax.integrations.hydra.instantiate_loggers

::: phijax.integrations.hydra.instantiate_trainer

::: phijax.integrations.hydra.build_trainer

::: phijax.integrations.hydra.configure_training

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
