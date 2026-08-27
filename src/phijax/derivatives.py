from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp

type ArgumentNumbers = int | Sequence[int]


def value_and_jacobian(
    function: Callable[..., Any],
    argnums: ArgumentNumbers,
) -> Callable[..., tuple[Any, tuple[Any, ...]]]:
    """Build a forward-mode value-and-Jacobian function for selected scalar arguments.

    The returned callable linearizes one primal evaluation and applies one JVP per selected argument. Each selected
    argument must be a scalar array. PINN equations can therefore split a point into scalar coordinates, call this
    helper at one point, and apply :func:`jax.vmap` outside it for collocation batches.

    Args:
        function: Pure function differentiated with respect to selected scalar positional arguments.
        argnums: Unique positional argument number or numbers, ordered like the requested Jacobian entries.

    Returns:
        Callable returning the primal output and one derivative PyTree per selected argument.

    Raises:
        TypeError: If `argnums` contains non-integer values or Boolean values.
        ValueError: If `argnums` is empty or duplicated, or a selected primal argument is not scalar.
    """
    resolved_argnums = _normalize_argnums(argnums)

    def wrapped(*args: Any) -> tuple[Any, tuple[Any, ...]]:
        """Evaluate the primal function and selected first derivatives.

        Args:
            *args: Primal positional arguments supplied to `function`.

        Returns:
            Primal output and derivative PyTrees aligned with `resolved_argnums`.
        """
        primals = tuple(args[index] for index in resolved_argnums)
        _validate_scalar_arguments(args, resolved_argnums)

        def selected_function(*selected: Any) -> Any:
            """Evaluate the function with selected arguments replaced.

            Args:
                *selected: Replacement scalar arguments aligned with `resolved_argnums`.

            Returns:
                Function output at the substituted arguments.
            """
            return function(*_substitute(args, resolved_argnums, selected))

        value, jvp_function = jax.linearize(selected_function, *primals)
        derivatives = []
        for selected_index in range(len(resolved_argnums)):
            tangents = tuple(
                jnp.ones_like(primal) if primal_index == selected_index else jnp.zeros_like(primal)
                for primal_index, primal in enumerate(primals)
            )
            derivatives.append(jvp_function(*tangents))
        return value, tuple(derivatives)

    return wrapped


def hessian_diagonal(
    function: Callable[..., Any],
    argnums: ArgumentNumbers,
) -> Callable[..., tuple[Any, ...]]:
    """Build forward-mode pure second derivatives for selected scalar arguments.

    Unlike a complete Hessian, this helper computes only `d²function / darg²` for each selected argument. It avoids
    tracing unused mixed derivatives in PDE operators such as one-dimensional diffusion and coordinate Laplacians.
    Every selected argument must be a scalar array.

    Args:
        function: Pure function differentiated with respect to selected scalar positional arguments.
        argnums: Unique positional argument number or numbers, ordered like the requested Hessian diagonal entries.

    Returns:
        Callable returning one pure-second-derivative PyTree per selected argument.

    Raises:
        TypeError: If `argnums` contains non-integer values or Boolean values.
        ValueError: If `argnums` is empty or duplicated, or a selected primal argument is not scalar.
    """
    resolved_argnums = _normalize_argnums(argnums)

    def wrapped(*args: Any) -> tuple[Any, ...]:
        """Evaluate selected Hessian-diagonal entries.

        Args:
            *args: Primal positional arguments supplied to `function`.

        Returns:
            Pure second derivatives aligned with `resolved_argnums`.
        """
        _validate_scalar_arguments(args, resolved_argnums)
        diagonal = []
        for index in resolved_argnums:

            def first_derivative(coordinate: jax.Array, *, coordinate_index: int = index) -> Any:
                """Evaluate one coordinate directional derivative.

                Args:
                    coordinate: Selected scalar coordinate.
                    coordinate_index: Static original positional argument number.

                Returns:
                    First derivative of the function output with respect to `coordinate`.
                """

                def selected_function(value: jax.Array) -> Any:
                    """Evaluate at one replacement scalar coordinate.

                    Args:
                        value: Replacement coordinate value.

                    Returns:
                        Function output at the replacement coordinate.
                    """
                    return function(*_substitute(args, (coordinate_index,), (value,)))

                return jax.jvp(selected_function, (coordinate,), (jnp.ones_like(coordinate),))[1]

            _, second_derivative = jax.jvp(
                first_derivative,
                (args[index],),
                (jnp.ones_like(args[index]),),
            )
            diagonal.append(second_derivative)
        return tuple(diagonal)

    return wrapped


def _normalize_argnums(argnums: ArgumentNumbers) -> tuple[int, ...]:
    """Validate and normalize selected positional argument numbers.

    Args:
        argnums: Candidate integer or sequence of integers.

    Returns:
        Non-empty unique immutable argument numbers.

    Raises:
        TypeError: If a value is not an integer or is Boolean.
        ValueError: If no values are supplied or a value is duplicated.
    """
    if isinstance(argnums, (bool, int)):
        resolved = (argnums,)
    elif isinstance(argnums, Sequence):
        resolved = tuple(argnums)
    else:
        raise TypeError("`argnums` must be an integer or a sequence of integers.")
    if any(isinstance(value, bool) or not isinstance(value, int) for value in resolved):
        raise TypeError("`argnums` must contain only integer positional argument numbers.")
    if not resolved or len(set(resolved)) != len(resolved):
        raise ValueError("`argnums` must be non-empty and unique.")
    return resolved


def _validate_scalar_arguments(args: tuple[Any, ...], argnums: tuple[int, ...]) -> None:
    """Validate that selected primal positional arguments are scalar.

    Args:
        args: Primal function arguments.
        argnums: Selected positional argument numbers.

    Raises:
        ValueError: If a selected primal has nonzero rank.
    """
    if any(jnp.ndim(args[index]) != 0 for index in argnums):
        raise ValueError("Selected derivative arguments must be scalar values.")


def _substitute(args: tuple[Any, ...], argnums: tuple[int, ...], values: tuple[Any, ...]) -> tuple[Any, ...]:
    """Replace selected positional arguments without mutating the input tuple.

    Args:
        args: Original positional arguments.
        argnums: Positions to replace.
        values: Replacement values aligned with `argnums`.

    Returns:
        Updated immutable argument tuple.
    """
    resolved = list(args)
    for index, value in zip(argnums, values, strict=True):
        resolved[index] = value
    return tuple(resolved)


__all__ = ["hessian_diagonal", "value_and_jacobian"]
