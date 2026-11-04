import sys
from typing import Any


def to_plain_container(value: Any, *, resolve: bool = True) -> Any:
    """Convert an OmegaConf container to ordinary Python containers.

    Args:
        value: Value that may be an OmegaConf configuration container.
        resolve: Whether to resolve OmegaConf interpolations during conversion.

    Returns:
        Plain Python containers for OmegaConf inputs, otherwise `value` unchanged.
    """
    # A config object can only exist after `omegaconf` was imported, so the optional `hydra` extra is never loaded here.
    omegaconf = sys.modules.get("omegaconf")
    if omegaconf is not None and omegaconf.OmegaConf.is_config(value):
        return omegaconf.OmegaConf.to_container(value, resolve=resolve)
    return value
