import random

import jax
import numpy as np
import pytest

from phijax.utils import resolve_seed, seed_everything


def test_resolve_seed_preserves_configured_value_and_generates_null_value(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify optional seeds resolve before global RNG initialization.

    Args:
        monkeypatch: Pytest helper replacing the operating-system random draw.
    """
    monkeypatch.setattr("phijax.utils.seeding.secrets.randbits", lambda bits: 314159 if bits == 32 else 0)

    assert resolve_seed(42) == 42
    assert resolve_seed(None) == 314159


@pytest.mark.parametrize("seed", [True, 1.5, "42"])
def test_resolve_seed_rejects_non_integer_values(seed: object) -> None:
    """Verify optional seed resolution rejects ambiguous configured values.

    Args:
        seed: Invalid configured seed value.
    """
    with pytest.raises(TypeError, match="integer"):
        resolve_seed(seed)  # type: ignore[arg-type]


def test_seed_everything_repeats_global_and_explicit_streams() -> None:
    """Verify Python, NumPy, and explicit JAX streams repeat for one process seed."""
    first_key = seed_everything(123, process_index=2)
    first_python = random.random()  # noqa: S311
    first_numpy = float(np.random.random())
    second_key = seed_everything(123, process_index=2)
    assert random.random() == first_python  # noqa: S311
    assert float(np.random.random()) == first_numpy
    assert bool(jax.numpy.array_equal(first_key, second_key))


def test_seed_everything_separates_distributed_processes() -> None:
    """Verify equal roots produce distinct JAX keys and host streams on different processes."""
    first_key = seed_everything(7, process_index=0)
    first_python = random.random()  # noqa: S311
    second_key = seed_everything(7, process_index=1)
    second_python = random.random()  # noqa: S311
    assert not bool(jax.numpy.array_equal(first_key, second_key))
    assert first_python != second_python


@pytest.mark.parametrize(
    ("seed", "process_index", "message"),
    [(-1, 0, "seed"), (2**32, 0, "seed"), (1, -1, "process_index")],
)
def test_seed_everything_rejects_invalid_values(seed: int, process_index: int, message: str) -> None:
    """Verify invalid root seeds and process indices fail explicitly.

    Args:
        seed: Invalid or accompanying root seed.
        process_index: Invalid or accompanying process index.
        message: Expected validation message fragment.
    """
    with pytest.raises(ValueError, match=message):
        seed_everything(seed, process_index=process_index)
