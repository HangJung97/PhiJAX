from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, ParamSpec, TypeVar

_RESIDUAL_NAMES_ATTRIBUTE = "__phijax_residual_names__"

P = ParamSpec("P")
R = TypeVar("R")


def residual_equation(*, names: Sequence[str]) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Attach ordered local residual names to an equation callable.

    The names describe the outer residual groups returned by the equation. They are local to the equation; a
    :class:`phijax.objectives.ResidualTerm` prefixes them with its configured `batch_key` when explicit loss names are
    omitted.

    Args:
        names: Non-empty unique local names ordered like the equation's outer residual groups.

    Returns:
        Decorator that preserves the equation callable and attaches its validated residual names.

    Raises:
        TypeError: If a name is not a string.
        ValueError: If names are empty, duplicated, or contain `/`.
    """
    resolved_names = _validate_residual_names(names)

    def decorate(equation: Callable[P, R]) -> Callable[P, R]:
        """Attach the already validated names to one equation callable.

        Args:
            equation: Equation callable whose residual groups follow `resolved_names`.

        Returns:
            Unchanged callable carrying static residual-name metadata.
        """
        setattr(equation, _RESIDUAL_NAMES_ATTRIBUTE, resolved_names)
        return equation

    return decorate


def get_residual_names(residual_fn: Callable[..., Any]) -> tuple[str, ...]:
    """Read static residual names from an equation or configured partial.

    Args:
        residual_fn: Decorated equation callable or a nested :class:`functools.partial` wrapping one.

    Returns:
        Ordered local residual names attached by :func:`residual_equation`.

    Raises:
        ValueError: If the callable does not define residual-name metadata.
    """
    equation: Callable[..., Any] = residual_fn
    while isinstance(equation, partial):
        equation = equation.func
    names = getattr(equation, _RESIDUAL_NAMES_ATTRIBUTE, None)
    if names is None:
        raise ValueError(
            "`residual_fn` does not define residual names. Decorate the equation with `residual_equation` or "
            "configure `names` explicitly."
        )
    return _validate_residual_names(names)


def _validate_residual_names(names: Sequence[str]) -> tuple[str, ...]:
    """Validate equation-local residual names.

    Args:
        names: Candidate names aligned with an equation's outer residual groups.

    Returns:
        Validated immutable residual names.

    Raises:
        TypeError: If a name is not a string.
        ValueError: If names are empty, duplicated, whitespace-only, or contain `/`.
    """
    resolved_names: list[str] = []
    for name in names:
        if not isinstance(name, str):
            raise TypeError("Equation residual names must be strings.")
        if not name or not name.strip():
            raise ValueError("Equation residual names must be non-empty.")
        if "/" in name:
            raise ValueError("Equation residual names must be local names without `/` separators.")
        resolved_names.append(name)
    if not resolved_names or len(set(resolved_names)) != len(resolved_names):
        raise ValueError("Equation residual names must be non-empty and unique.")
    return tuple(resolved_names)


__all__ = ["get_residual_names", "residual_equation"]
