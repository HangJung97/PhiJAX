from collections.abc import Callable, Iterable, Sequence
from typing import Any

import optax
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from phijax.balancers import LossBalancer
from phijax.callbacks import Callback
from phijax.core import BasePhiModule
from phijax.data import PhiDataModule
from phijax.models import ModelFactory
from phijax.objectives import Objective
from phijax.training.loggers import ExperimentLogger, LoggerCollection
from phijax.training.trainer import Trainer


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
    """Instantiate and validate callback configurations.

    Args:
        config: Optional callback configuration mapping.

    Returns:
        Ordered callback instances.

    Raises:
        TypeError: If an entry is not a mapping or does not instantiate :class:`Callback`.
        ValueError: If an entry is null, lacks `_target_`, or uses the removed `enabled` option.
    """
    if config is None:
        return ()
    entries = (("callback", config),) if "_target_" in config else tuple(config.items())
    callbacks: list[Callback] = []
    for name, node in entries:
        if node is None:
            raise ValueError(f"Callback `{name}` is null; omit the entry to disable it.")
        if not isinstance(node, DictConfig):
            raise TypeError(f"Callback `{name}` must be a Hydra configuration mapping.")
        if "enabled" in node:
            raise ValueError(f"Callback `{name}` uses the removed `enabled` option; omit the entry to disable it.")
        if "_target_" not in node:
            raise ValueError(f"Callback `{name}` must define `_target_`.")
        callback = instantiate(node)
        if not isinstance(callback, Callback):
            raise TypeError(f"Callback `{name}` must instantiate `Callback`.")
        callbacks.append(callback)
    return tuple(callbacks)


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
    trainer = instantiate(config, callbacks=tuple(callbacks), logger=False)
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


def instantiate_data_module(config: DictConfig) -> PhiDataModule:
    """Instantiate one configured DataModule without preparing a runtime stage.

    Args:
        config: Hydra-instantiable DataModule configuration.

    Returns:
        Unprepared DataModule whose stage lifecycle is owned by :class:`phijax.Trainer`.

    Raises:
        TypeError: If `config` does not instantiate :class:`PhiDataModule`.
    """
    data_module = instantiate(config)
    if not isinstance(data_module, PhiDataModule):
        raise TypeError("The `data` config must instantiate `PhiDataModule`.")
    return data_module


def instantiate_model_factory(config: DictConfig) -> ModelFactory:
    """Instantiate a lazy model factory from one Hydra target.

    Args:
        config: Hydra-instantiable model-builder configuration.

    Returns:
        Callable retaining architecture options while leaving key, normalization, and precision unbound.

    Raises:
        TypeError: If the configured target cannot produce a callable factory.
    """
    factory = instantiate(config, _partial_=True)
    if not isinstance(factory, Callable):
        raise TypeError("The configured model target must produce a callable factory.")
    return factory


def instantiate_module(
    config: DictConfig,
    model: ModelFactory,
    objective: Objective,
    *,
    name: str,
) -> BasePhiModule:
    """Instantiate a configurable application module blueprint.

    Args:
        config: Hydra-instantiable module configuration.
        model: Lazy initialized-model factory.
        objective: Instantiated objective producing named scalar losses.
        name: Human-readable application identity.

    Returns:
        Module implementing the trainer-facing lifecycle and computation contract.

    Raises:
        TypeError: If `config` does not instantiate :class:`BasePhiModule`.
    """
    module = instantiate(
        config,
        model=model,
        objective=objective,
        name=name,
    )
    if not isinstance(module, BasePhiModule):
        raise TypeError("The module config must instantiate `BasePhiModule`.")
    return module


def instantiate_balancer(config: DictConfig, loss_names: Sequence[str]) -> Any:
    """Instantiate and validate one loss balancer.

    Args:
        config: Directly Hydra-instantiable balancer configuration containing `_target_` and constructor options.
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
    trainer.set_logger(instantiate_loggers(config.get("logger"), trainer))
    return trainer


__all__ = [
    "build_trainer",
    "instantiate_balancer",
    "instantiate_callbacks",
    "instantiate_data_module",
    "instantiate_enabled",
    "instantiate_loggers",
    "instantiate_model_factory",
    "instantiate_module",
    "instantiate_objective",
    "instantiate_optimizer",
    "instantiate_trainer",
]
