from collections.abc import Callable, Sequence
from functools import partial
from typing import Any, ParamSpec, TypeVar, cast

from phijax.types import ResidualStream

_RESIDUAL_NAMES_ATTRIBUTE = "__phijax_residual_names__"
_DEFAULT_NTK_STREAM_ATTRIBUTE = "__phijax_default_ntk_stream__"

P = ParamSpec("P")
R = TypeVar("R")


def residual_equation(
    *,
    names: Sequence[str],
    default_ntk_stream: ResidualStream = "residual",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Attach ordered local residual names and balancing metadata to an equation callable.

    The names describe the outer residual groups returned by the equation. They are local to the equation; a
    :class:`phijax.objectives.ResidualTerm` prefixes them with its configured `batch_key` when explicit loss names are
    omitted.

    Args:
        names: Non-empty unique local names ordered like the equation's outer residual groups.
        default_ntk_stream: Stream used by derivative-based balancing when a term does not override it.

    Returns:
        Decorator that preserves the equation callable and attaches its validated residual names.

    Raises:
        TypeError: If a name is not a string.
        ValueError: If names are empty, duplicated, contain `/`, or the stream is unsupported.
    """
    resolved_names = _validate_residual_names(names)
    resolved_stream = _validate_ntk_stream(default_ntk_stream)

    def decorate(equation: Callable[P, R]) -> Callable[P, R]:
        """Attach the already validated names to one equation callable.

        Args:
            equation: Equation callable whose residual groups follow `resolved_names`.

        Returns:
            Unchanged callable carrying static residual-name metadata.
        """
        setattr(equation, _RESIDUAL_NAMES_ATTRIBUTE, resolved_names)
        setattr(equation, _DEFAULT_NTK_STREAM_ATTRIBUTE, resolved_stream)
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


def get_default_ntk_stream(residual_fn: Callable[..., Any]) -> ResidualStream:
    """Read the preferred balancing stream from an equation or configured partial.

    Undecorated callables default to the residual stream so explicit :class:`phijax.ResidualTerm` construction remains
    usable when custom names are supplied.

    Args:
        residual_fn: Equation callable or a nested :class:`functools.partial` wrapping one.

    Returns:
        Decorated default stream, or `"residual"` when no stream metadata is attached.
    """
    equation: Callable[..., Any] = residual_fn
    while isinstance(equation, partial):
        equation = equation.func
    return _validate_ntk_stream(getattr(equation, _DEFAULT_NTK_STREAM_ATTRIBUTE, "residual"))


def _validate_ntk_stream(stream: str) -> ResidualStream:
    """Validate a derivative-balancing stream name.

    Args:
        stream: Candidate equation stream.

    Returns:
        Validated `"residual"` or `"output"` stream.

    Raises:
        ValueError: If `stream` is unsupported.
    """
    if stream not in ("residual", "output"):
        raise ValueError("`default_ntk_stream` must be either 'residual' or 'output'.")
    return cast(ResidualStream, stream)


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


__all__ = ["get_default_ntk_stream", "get_residual_names", "residual_equation"]
