from types import SimpleNamespace

import jax
import pytest

from tests.helpers.run import RunIf


def test_runif_skips_when_too_few_gpus_are_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the decorator skips a test whose GPU requirement is unmet."""
    monkeypatch.setattr(jax, "local_devices", lambda: [SimpleNamespace(platform="cpu")])

    marker = RunIf(min_gpus=1)

    assert marker.mark.name == "skipif"
    assert marker.mark.kwargs["condition"] is True
    assert marker.mark.kwargs["reason"] == "Requires: [GPUs>=1]"


def test_runif_runs_when_enough_gpus_are_visible(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify the decorator runs a test whose GPU requirement is satisfied."""
    monkeypatch.setattr(
        jax,
        "local_devices",
        lambda: [SimpleNamespace(platform="gpu"), SimpleNamespace(platform="cpu")],
    )

    marker = RunIf(min_gpus=1)

    assert marker.mark.kwargs["condition"] is False


def test_runif_skips_when_jax_cannot_initialize_a_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify a JAX backend initialization failure is treated as unavailable."""

    def raise_backend_error() -> None:
        """Simulate JAX failing to initialize its configured backend.

        Raises:
            RuntimeError: Always, to represent an unavailable backend.
        """
        msg = "backend unavailable"
        raise RuntimeError(msg)

    monkeypatch.setattr(jax, "local_devices", raise_backend_error)

    marker = RunIf(min_gpus=1)

    assert marker.mark.kwargs["condition"] is True


def test_runif_rejects_a_negative_gpu_requirement() -> None:
    """Verify invalid negative GPU requirements fail during collection setup."""
    with pytest.raises(ValueError, match="min_gpus must be non-negative"):
        RunIf(min_gpus=-1)
