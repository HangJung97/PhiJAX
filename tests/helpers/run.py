from typing import Any

import jax
import pytest


def _local_gpu_count() -> int:
    """Count GPU devices visible to the current JAX process.

    Returns:
        Number of local JAX devices whose platform is `gpu`. Returns zero when
        JAX cannot initialize an available backend.
    """
    try:
        return sum(device.platform == "gpu" for device in jax.local_devices())
    except RuntimeError:
        return 0


class RunIf:
    """Create pytest decorators for tests with runtime device requirements.

    Examples:
        Require one JAX-visible GPU for a test::

            @RunIf(min_gpus=1)
            def test_gpu_operation() -> None:
                ...
    """

    def __new__(
        cls,
        min_gpus: int = 0,
        **kwargs: Any,
    ) -> pytest.MarkDecorator:
        """Create a conditional pytest skip decorator.

        Args:
            min_gpus: Minimum number of local JAX GPU devices required to run
                the decorated test.
            **kwargs: Additional keyword arguments forwarded to
                :func:`pytest.mark.skipif`.

        Returns:
            A pytest decorator that skips the test when fewer than `min_gpus`
            devices are visible.

        Raises:
            ValueError: If `min_gpus` is negative.
        """
        del cls
        if min_gpus < 0:
            msg = "min_gpus must be non-negative"
            raise ValueError(msg)

        insufficient_gpus = _local_gpu_count() < min_gpus
        reason = f"Requires: [GPUs>={min_gpus}]"
        return pytest.mark.skipif(condition=insufficient_gpus, reason=reason, **kwargs)
