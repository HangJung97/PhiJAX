# Utilities

Utilities cover reproducibility, rank-aware Python logging, configuration display, task cleanup, shape normalization,
and conversion at JAX/OmegaConf boundaries.

## Reproducibility

`resolve_seed(None)` selects a random unsigned 32-bit seed. The training entrypoint writes that concrete value back to
the composed config and logs it before initialization. `seed_everything` seeds Python and NumPy and returns a JAX key
folded with the current process index.

::: phijax.utils.resolve_seed

::: phijax.utils.seed_everything

## Python logging

::: phijax.utils.RankedLogger

::: phijax.utils.get_colorlogger

::: phijax.utils.pad_keys

## Configuration and task lifecycle

`pre_hydra_routine` must run before a decorated Hydra entrypoint so project-root discovery, environment defaults, and
OmegaConf resolver registration happen before composition. `task_wrapper` logs failures and always runs registered
finalizers, including optional W&B shutdown.

::: phijax.utils.pre_hydra_routine

::: phijax.utils.extras

::: phijax.utils.print_config_tree

::: phijax.utils.task_wrapper

::: phijax.utils.register_task_finalizer

## Conversion and shapes

::: phijax.utils.as_numpy

::: phijax.utils.to_plain_container

::: phijax.utils.as_tuple
