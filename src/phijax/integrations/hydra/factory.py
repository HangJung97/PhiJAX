from collections.abc import Iterable, Sequence
from typing import Any

import jax
import optax
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from phijax.balancers import LossBalancer
from phijax.callbacks import Callback
from phijax.data import DataStage, PhiDataModule
from phijax.models import InitializedModel
from phijax.module import BasePhiModule
from phijax.objectives import Objective
from phijax.training.loggers import ExperimentLogger, LoggerCollection
from phijax.training.precision import PrecisionPolicy
from phijax.training.trainer import Trainer
from phijax.types import ModelApply, ModelSummaryFunction


def instantiate_enabled(config: DictConfig | None) -> tuple[Any, ...]:
    """Instantiate enabled Hydra service entries from one mapping.

    A mapping containing `_target_` is treated as one service. Otherwise each child entry is considered independently.
    Entries that are `None` or set `enabled: false` are omitted; the orchestration-only `enabled` key is removed before
    Hydra calls the target constructor.

    Args:
        config: Optional single-service config or mapping of named service configs.

    Returns:
        Ordered tuple of instantiated services.
    """
    if config is None:
        return ()
    nodes: Iterable[Any] = (config,) if "_target_" in config else config.values()
    instances: list[Any] = []
    for node in nodes:
        if node is None or not isinstance(node, DictConfig):
            continue
        raw = OmegaConf.to_container(node, resolve=True)
        if not isinstance(raw, dict) or "_target_" not in raw or not raw.get("enabled", True):
            continue
        raw.pop("enabled", None)
        instances.append(instantiate(OmegaConf.create(raw)))
    return tuple(instances)


def instantiate_callbacks(config: DictConfig | None) -> tuple[Callback, ...]:
    """Instantiate and validate enabled callback configurations.

    Args:
        config: Optional callback configuration mapping.

    Returns:
        Ordered callback instances.

    Raises:
        TypeError: If an enabled entry does not instantiate :class:`Callback`.
    """
    callbacks = instantiate_enabled(config)
    if any(not isinstance(callback, Callback) for callback in callbacks):
        raise TypeError("Every enabled callback config must instantiate `Callback`.")
    return callbacks


def instantiate_trainer(config: DictConfig, callbacks: Sequence[Callback] = ()) -> Trainer:
    """Instantiate a trainer with already constructed callbacks.

    Args:
        config: Hydra-instantiable trainer configuration.
        callbacks: Ordered callback instances owned by the trainer.

    Returns:
        Instantiated trainer with an empty logger collection.

    Raises:
        TypeError: If `config` does not instantiate :class:`Trainer`.
    """
    trainer = instantiate(config, callbacks=tuple(callbacks), loggers=())
    if not isinstance(trainer, Trainer):
        raise TypeError("The `trainer` config must instantiate `Trainer`.")
    return trainer


def instantiate_loggers(config: DictConfig | None, trainer: Trainer) -> LoggerCollection:
    """Instantiate external loggers for the trainer's global process.

    Args:
        config: Optional logger configuration mapping.
        trainer: Trainer supplying distributed rank ownership.

    Returns:
        Validated logger collection, empty on nonzero processes.

    Raises:
        TypeError: If an enabled entry does not instantiate :class:`ExperimentLogger`.
    """
    loggers = instantiate_enabled(config) if trainer.strategy.is_global_zero else ()
    if any(not isinstance(logger, ExperimentLogger) for logger in loggers):
        raise TypeError("Every enabled logger config must instantiate `ExperimentLogger`.")
    return LoggerCollection(loggers)


def instantiate_objective(config: DictConfig) -> Objective:
    """Instantiate and validate one objective configuration.

    Args:
        config: Hydra-instantiable objective configuration.

    Returns:
        Objective implementing the generic scalar-loss contract.

    Raises:
        TypeError: If `config` does not instantiate :class:`Objective`.
    """
    objective = instantiate(config)
    if not isinstance(objective, Objective):
        raise TypeError("The objective config must instantiate `Objective`.")
    return objective


def instantiate_data_module(config: DictConfig, stage: DataStage) -> PhiDataModule:
    """Instantiate and set up one configured DataModule stage.

    Args:
        config: Hydra-instantiable DataModule configuration.
        stage: Initial `fit` or `predict` stage.

    Returns:
        Prepared DataModule with pools for `stage`.

    Raises:
        TypeError: If `config` does not instantiate :class:`PhiDataModule`.
    """
    data_module = instantiate(config)
    if not isinstance(data_module, PhiDataModule):
        raise TypeError("The `data` config must instantiate `PhiDataModule`.")
    data_module.prepare_stage(stage)
    return data_module


def instantiate_model(
    config: DictConfig,
    precision: PrecisionPolicy,
    data_module: PhiDataModule,
    key: jax.Array,
) -> InitializedModel:
    """Instantiate a configured network and its explicit model state.

    Args:
        config: Hydra-instantiable network configuration.
        precision: Trainer precision policy applied during model construction.
        data_module: Prepared DataModule supplying input normalization statistics.
        key: Explicit model initialization key.

    Returns:
        Validated pure model application, explicit state, and optional summary.

    Raises:
        TypeError: If the configured factory does not return :class:`~phijax.models.InitializedModel`.
    """
    mean, std = data_module.input_statistics()
    initialized = instantiate(
        config,
        key=key,
        precision=precision.mode,
        input_mean=mean,
        input_std=std,
    )
    if not isinstance(initialized, InitializedModel):
        raise TypeError("The configured model factory must return `InitializedModel`.")
    return initialized


def instantiate_module(
    config: DictConfig,
    model_apply: ModelApply,
    objective: Objective,
    *,
    name: str,
    model_summary: ModelSummaryFunction | None = None,
) -> BasePhiModule:
    """Instantiate a configurable application module around runtime model objects.

    Args:
        config: Hydra-instantiable module configuration.
        model_apply: Pure explicit-state network application callable.
        objective: Instantiated objective producing named scalar losses.
        name: Human-readable application identity.
        model_summary: Optional network-summary callable.

    Returns:
        Module implementing the trainer-facing lifecycle and computation contract.

    Raises:
        TypeError: If `config` does not instantiate :class:`BasePhiModule`.
    """
    module = instantiate(
        config,
        model_apply=model_apply,
        objective=objective,
        name=name,
        model_summary=model_summary,
    )
    if not isinstance(module, BasePhiModule):
        raise TypeError("The module config must instantiate `BasePhiModule`.")
    return module


def instantiate_balancer(config: DictConfig, loss_names: Sequence[str]) -> Any:
    """Instantiate and validate one loss balancer.

    Args:
        config: Hydra-instantiable balancer factory configuration.
        loss_names: Stable objective loss names injected into the balancer.

    Returns:
        Loss balancer exposing initialization and scalar combination operations.

    Raises:
        TypeError: If the constructed object does not implement the balancer contract.
    """
    balancer = instantiate(config, loss_names=tuple(loss_names))
    if not isinstance(balancer, LossBalancer):
        raise TypeError("The configured balancer must implement `LossBalancer`.")
    return balancer


def instantiate_optimizer(config: DictConfig) -> optax.GradientTransformation:
    """Instantiate and validate one Optax optimizer configuration.

    Args:
        config: Hydra-instantiable optimizer configuration.

    Returns:
        Optax gradient transformation.

    Raises:
        TypeError: If the constructed object is not an Optax gradient transformation.
    """
    optimizer = instantiate(config)
    if not isinstance(optimizer, optax.GradientTransformation):
        raise TypeError("The configured optimizer must be an Optax gradient transformation.")
    return optimizer


def build_trainer(config: DictConfig) -> Trainer:
    """Build a trainer from root callback, trainer, and logger groups.

    Args:
        config: Root Hydra configuration containing `callbacks`, `trainer`, and `logger`.

    Returns:
        Fully configured trainer.
    """
    callbacks = instantiate_callbacks(config.get("callbacks"))
    trainer = instantiate_trainer(config.trainer, callbacks)
    trainer.logger = instantiate_loggers(config.get("logger"), trainer)
    return trainer


__all__ = [
    "build_trainer",
    "instantiate_balancer",
    "instantiate_callbacks",
    "instantiate_data_module",
    "instantiate_enabled",
    "instantiate_loggers",
    "instantiate_model",
    "instantiate_module",
    "instantiate_objective",
    "instantiate_optimizer",
    "instantiate_trainer",
]
