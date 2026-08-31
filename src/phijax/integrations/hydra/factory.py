from collections.abc import Callable, Sequence
from typing import Any, cast

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


def _service_entries(config: DictConfig | None, service_name: str) -> tuple[tuple[str, DictConfig], ...]:
    """Validate and return configured Hydra service entries.

    Args:
        config: Optional single-service config or mapping of named service configs.
        service_name: User-facing service type used in validation errors.

    Returns:
        Ordered service names and Hydra configuration mappings.

    Raises:
        TypeError: If an entry is not a Hydra configuration mapping.
        ValueError: If an entry is null, lacks `_target_`, or uses the removed `enabled` option.
    """
    if config is None:
        return ()
    entries = ((service_name.lower(), config),) if "_target_" in config else tuple(config.items())
    validated: list[tuple[str, DictConfig]] = []
    for name, node in entries:
        if node is None:
            raise ValueError(f"{service_name} `{name}` is null; omit the entry to disable it.")
        if not isinstance(node, DictConfig):
            raise TypeError(f"{service_name} `{name}` must be a Hydra configuration mapping.")
        if "enabled" in node:
            raise ValueError(
                f"{service_name} `{name}` uses the removed `enabled` option; omit the entry to disable it."
            )
        if "_target_" not in node:
            raise ValueError(f"{service_name} `{name}` must define `_target_`.")
        validated.append((str(name), node))
    return tuple(validated)


def instantiate_callbacks(config: DictConfig | None) -> tuple[Callback, ...]:
    """Instantiate and validate callback configurations.

    Args:
        config: Optional callback configuration mapping.

    Returns:
        Ordered callback instances.

    Raises:
        TypeError: If an entry is invalid or does not instantiate :class:`Callback`.
        ValueError: If an entry is null, lacks `_target_`, or uses `enabled`.
    """
    callbacks: list[Callback] = []
    for name, node in _service_entries(config, "Callback"):
        callback = instantiate(node)
        if not isinstance(callback, Callback):
            raise TypeError(f"Callback `{name}` must instantiate `Callback`.")
        callbacks.append(callback)
    return tuple(callbacks)


def instantiate_trainer(
    config: DictConfig,
    callbacks: Sequence[Callback] = (),
    logger: bool | ExperimentLogger | Sequence[ExperimentLogger] | None = False,
) -> Trainer:
    """Instantiate a trainer with already constructed host services.

    Args:
        config: Hydra-instantiable trainer configuration.
        callbacks: Ordered callback instances owned by the trainer.
        logger: Default logger flag, one backend, several backends, or `None` to disable logging.

    Returns:
        Instantiated trainer owning the configured callbacks and loggers.

    Raises:
        TypeError: If `config` does not instantiate :class:`Trainer`.
    """
    trainer = instantiate(config, callbacks=tuple(callbacks), logger=logger)
    if not isinstance(trainer, Trainer):
        raise TypeError("The `trainer` config must instantiate `Trainer`.")
    return trainer


def instantiate_loggers(config: DictConfig | None) -> LoggerCollection:
    """Instantiate external loggers without acquiring their runtime resources.

    Args:
        config: Optional logger configuration mapping.

    Returns:
        Validated logger collection in configuration order.

    Raises:
        TypeError: If an entry is invalid or does not instantiate :class:`ExperimentLogger`.
        ValueError: If an entry is null, lacks `_target_`, or uses `enabled`.
    """
    entries = _service_entries(config, "Logger")
    loggers: list[ExperimentLogger] = []
    for name, node in entries:
        logger = instantiate(node)
        if not isinstance(logger, ExperimentLogger):
            raise TypeError(f"Logger `{name}` must instantiate `ExperimentLogger`.")
        loggers.append(logger)
    return LoggerCollection(loggers)


def to_hyperparameters(config: DictConfig, *, resolve: bool = False) -> dict[str, Any]:
    """Convert a composed Hydra config into logger-ready plain containers.

    The Trainer owns the actual rank-safe logger call when the returned mapping is passed to
    :meth:`phijax.Trainer.fit` or :meth:`phijax.Trainer.fit_state`.

    Args:
        config: Composed root Hydra configuration.
        resolve: Whether to resolve OmegaConf interpolations before conversion.

    Returns:
        Plain nested mapping accepted by the Trainer's `hyperparameters` argument.

    Raises:
        TypeError: If `config` does not contain a mapping at its root.
    """
    parameters = OmegaConf.to_container(config, resolve=resolve)
    if not isinstance(parameters, dict):
        raise TypeError("The root hyperparameter configuration must be a mapping.")
    return cast(dict[str, Any], parameters)


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
    loggers = instantiate_loggers(config.get("logger"))
    return instantiate_trainer(config.trainer, callbacks, logger=loggers)


__all__ = [
    "build_trainer",
    "instantiate_balancer",
    "instantiate_callbacks",
    "instantiate_data_module",
    "instantiate_loggers",
    "instantiate_model_factory",
    "instantiate_module",
    "instantiate_objective",
    "instantiate_optimizer",
    "instantiate_trainer",
    "to_hyperparameters",
]
