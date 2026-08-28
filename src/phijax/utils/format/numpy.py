from importlib import import_module
from importlib.util import find_spec
from typing import Any

import numpy as np


def as_numpy(value: Any) -> np.ndarray:
    """Convert an array-like value to a host NumPy array.

    JAX is imported only when it is installed and conversion is requested. This keeps configuration and evaluation
    utilities importable without a JAX runtime while ensuring device arrays are explicitly transferred to the host.

    Args:
        value: Array-like value to convert. JAX arrays may reside on any supported device.

    Returns:
        Host NumPy view or copy of `value`. JAX `bfloat16` arrays are promoted to `float32` because NumPy does not
        provide a native `bfloat16` dtype.
    """
    if isinstance(value, np.ndarray):
        return value

    host_value = value
    if find_spec("jax") is not None:
        jax = import_module("jax")
        host_value = jax.device_get(value)

    if str(getattr(host_value, "dtype", "")) == "bfloat16":
        host_value = host_value.astype(np.float32)
    return np.asarray(host_value)
