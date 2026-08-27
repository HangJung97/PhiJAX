import jax.numpy as jnp
import numpy as np
from omegaconf import OmegaConf

from phijax.utils.format import as_numpy, to_plain_container


def test_as_numpy_transfers_jax_arrays_and_promotes_bfloat16() -> None:
    """Verify JAX device arrays become host arrays with NumPy-compatible dtypes."""
    value = jnp.asarray([1.0, 2.0], dtype=jnp.bfloat16)
    converted = as_numpy(value)
    assert isinstance(converted, np.ndarray)
    assert converted.dtype == np.float32
    np.testing.assert_array_equal(converted, [1.0, 2.0])


def test_as_numpy_preserves_numpy_arrays() -> None:
    """Verify an existing NumPy array is returned without an unnecessary copy."""
    value = np.asarray([1, 2])
    assert as_numpy(value) is value


def test_to_plain_container_resolves_omegaconf_values_only_when_requested() -> None:
    """Verify OmegaConf conversion and interpolation-resolution control."""
    config = OmegaConf.create({"value": 3, "copy": "${value}"})
    unresolved = to_plain_container(config, resolve=False)
    resolved = to_plain_container(config)
    assert unresolved == {"value": 3, "copy": "${value}"}
    assert resolved == {"value": 3, "copy": 3}
    marker = object()
    assert to_plain_container(marker) is marker
