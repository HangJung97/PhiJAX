from typing import Any

from omegaconf import OmegaConf


def to_plain_container(value: Any, *, resolve: bool = True) -> Any:
    """Convert an OmegaConf container to ordinary Python containers.

    Args:
        value: Value that may be an OmegaConf configuration container.
        resolve: Whether to resolve OmegaConf interpolations during conversion.

    Returns:
        Plain Python containers for OmegaConf inputs, otherwise `value` unchanged.
    """
    if OmegaConf.is_config(value):
        return OmegaConf.to_container(value, resolve=resolve)
    return value
