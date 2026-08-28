from typing import Any, cast

import optax
from omegaconf import DictConfig, OmegaConf

from phijax.balancers import AdaptiveBalancer
from phijax.module import BasePhiModule
from phijax.training.plans import BalancerUpdateSchedule, TrainingPlan
from phijax.training.trainer import Trainer


def configure_training(
    config: DictConfig,
    trainer: Trainer,
    module: BasePhiModule,
    balancer: Any,
    optimizer: optax.GradientTransformation,
) -> TrainingPlan:
    """Configure a compiled training plan from instantiated objects.

    Args:
        config: Model configuration containing objective terms and optional adaptive-balancer scheduling.
        trainer: Trainer compiling the update and preparing sampler placement.
        module: Instantiated application module.
        balancer: Independently instantiated loss balancer.
        optimizer: Instantiated Optax gradient transformation.

    Returns:
        Compiled step, required batch keys, and optional adaptive-balancer schedule.

    Raises:
        TypeError: If adaptive update configuration is paired with an incompatible balancer.
        ValueError: If adaptive scheduling values are invalid.
    """
    train_step = trainer.compile_train_step(module, balancer, optimizer)
    batch_keys = tuple(dict.fromkeys(str(term.batch_key) for term in config.objective.terms.values()))
    update_config = config.balancer.get("update")
    if update_config is None:
        return TrainingPlan(train_step, batch_keys)
    if not isinstance(balancer, AdaptiveBalancer):
        raise TypeError("A non-null balancer `update` config requires an `AdaptiveBalancer` implementation.")
    resolved_update = OmegaConf.to_container(update_config, resolve=True)
    if not isinstance(resolved_update, dict):
        raise TypeError("`model.balancer.update` must resolve to a mapping.")
    every_n_steps = resolved_update.pop("every_n_steps")
    if isinstance(every_n_steps, bool) or not isinstance(every_n_steps, int):
        raise TypeError("`model.balancer.update.every_n_steps` must be an integer.")
    skip_first_step = resolved_update.pop("skip_first_step", True)
    if not isinstance(skip_first_step, bool):
        raise TypeError("`model.balancer.update.skip_first_step` must be Boolean.")
    update_plan = balancer.build_update_plan(module, batch_keys, cast(dict[str, Any], resolved_update))
    schedule = BalancerUpdateSchedule(
        update_plan,
        every_n_steps,
        skip_first_step=skip_first_step,
    )
    return TrainingPlan(train_step, batch_keys, schedule)


__all__ = ["configure_training"]
