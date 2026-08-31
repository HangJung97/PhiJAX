from __future__ import annotations

from typing import TYPE_CHECKING, cast

import optax

from phijax.balancers import AdaptiveBalancer, LossBalancer
from phijax.core import BasePhiModule
from phijax.training.plans import TrainingPlan

if TYPE_CHECKING:
    from phijax.training.trainer import Trainer


def build_training_plan(
    trainer: Trainer,
    module: BasePhiModule,
    balancer: LossBalancer,
    optimizer: optax.GradientTransformation,
) -> TrainingPlan:
    """Resolve a configuration-independent compiled training plan.

    Args:
        trainer: Trainer supplying precision-aware step compilation.
        module: Bound module exposing stable loss and batch names.
        balancer: Functional loss balancer aligned with the module losses.
        optimizer: Optax gradient transformation used by the update.

    Returns:
        Compiled optimizer step, inferred batch routing, and optional adaptive update plan.

    Raises:
        TypeError: If the balancer does not satisfy :class:`phijax.LossBalancer`.
        ValueError: If loss names differ or batch names are unavailable.
    """
    if not isinstance(balancer, LossBalancer):
        raise TypeError("`balancer` must implement `LossBalancer`.")
    if tuple(balancer.loss_names) != tuple(module.loss_names):
        raise ValueError("Balancer loss names must exactly match the module loss-name ordering.")
    batch_keys = tuple(module.batch_keys)
    if not batch_keys:
        raise ValueError("The concise training path requires the module to expose at least one batch key.")
    adaptive = isinstance(balancer, AdaptiveBalancer)
    update_plan = cast(AdaptiveBalancer, balancer).build_update_plan(module, batch_keys) if adaptive else None
    return TrainingPlan(trainer.compile_train_step(module, balancer, optimizer), batch_keys, update_plan)


__all__ = ["build_training_plan"]
