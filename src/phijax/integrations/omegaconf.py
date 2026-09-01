import importlib
import logging
import math
import operator
from collections.abc import Callable
from typing import Any

from omegaconf import OmegaConf

log = logging.getLogger(__name__)


def register_omegaconf_resolvers() -> None:
    """Register PhiJAX's general-purpose OmegaConf resolvers idempotently.

    The registry exposes configuration assertions, standard-library operators, ternary selection, importable callable
    invocation, tuple construction, Hydra target names, and public constants or callables from :mod:`math`. Resolver
    registration modifies OmegaConf process-global state, so this function is safe to call repeatedly before Hydra
    composition.

    Examples:
        `${math:pi}` resolves to :data:`math.pi`, while `${op:truediv,0.01,${math:pi}}` resolves to `0.01 / math.pi`.
    """
    _register_new_resolver_once("assert", _assert_configuration)
    _register_new_resolver_once("op", _resolve_operator)
    _register_new_resolver_once("op.ternary", _resolve_ternary)
    _register_new_resolver_once("call", _resolve_callable)
    _register_new_resolver_once("tuple", _resolve_tuple)
    _register_new_resolver_once("target.name", _resolve_target_name)
    _register_new_resolver_once("math", _resolve_math)


def _register_new_resolver_once(name: str, resolver: Callable[..., Any]) -> None:
    """Register one OmegaConf resolver unless the name is already present.

    Args:
        name: Global OmegaConf resolver name.
        resolver: Callable evaluating interpolation arguments.
    """
    if not OmegaConf.has_resolver(name):
        OmegaConf.register_new_resolver(name, resolver)


def _assert_configuration(condition: bool, throw_on_fail: bool = True) -> bool:
    """Validate one deferred configuration condition.

    Args:
        condition: Resolved condition that must be truthy.
        throw_on_fail: Whether a false condition raises instead of logging a warning.

    Returns:
        The resolved truth value.

    Raises:
        AssertionError: If `condition` is false and `throw_on_fail` is true.
    """
    if condition:
        return True
    message = "Hydra configuration assertion failed."
    if throw_on_fail:
        raise AssertionError(message)
    log.warning(message)
    return False


def _resolve_operator(name: str, *args: Any) -> Any:
    """Apply one public callable from Python's :mod:`operator` module.

    Args:
        name: Public operator function name such as `add`, `truediv`, or `eq`.
        *args: Values passed positionally to the selected operator.

    Returns:
        Result returned by the selected operator.

    Raises:
        ValueError: If `name` does not identify a public callable operator.
    """
    operation = getattr(operator, name, None) if not name.startswith("_") else None
    if not callable(operation):
        raise ValueError(f"Unknown public operator `{name}`.")
    return operation(*args)


def _resolve_ternary(condition: Any, true_value: Any, false_value: Any) -> Any:
    """Select one of two configuration values from a resolved condition.

    Args:
        condition: Value interpreted by Python truthiness rules.
        true_value: Value returned when `condition` is truthy.
        false_value: Value returned when `condition` is falsey.

    Returns:
        `true_value` or `false_value` according to `condition`.
    """
    return true_value if condition else false_value


def _resolve_callable(dotpath: str, *args: Any) -> Any:
    """Import and invoke a callable using a dotted Python path.

    Args:
        dotpath: Import path whose final component names the callable.
        *args: Values passed positionally to the imported callable.

    Returns:
        Value returned by the imported callable.

    Raises:
        TypeError: If `dotpath` resolves to a non-callable object.
    """
    target = import_from_module(dotpath)
    if not callable(target):
        raise TypeError(f"Imported config target `{dotpath}` is not callable.")
    return target(*args)


def _resolve_tuple(*args: Any) -> tuple[Any, ...]:
    """Build a tuple from resolved interpolation arguments.

    Args:
        *args: Resolved values retained in declaration order.

    Returns:
        Tuple containing `args`.
    """
    return tuple(args)


def _resolve_target_name(dotpath: str) -> str:
    """Extract the callable name from a dotted Hydra target.

    Args:
        dotpath: Fully qualified Hydra `_target_` value.

    Returns:
        Final target component with its original spelling.

    Raises:
        TypeError: If `dotpath` is not a string.
        ValueError: If `dotpath` does not contain a module and target name.
    """
    if not isinstance(dotpath, str):
        raise TypeError("A Hydra target path must be a string.")
    if "." not in dotpath or not dotpath.rsplit(".", 1)[1]:
        raise ValueError("A Hydra target path must contain a module and target name.")
    return dotpath.rsplit(".", 1)[1]


def _resolve_math(name: str, *args: Any) -> Any:
    """Resolve one public constant or invoke one public callable from :mod:`math`.

    Args:
        name: Public :mod:`math` attribute such as `pi`, `sqrt`, or `sin`.
        *args: Positional arguments for callable attributes; constants accept none.

    Returns:
        Selected constant or result of the selected mathematical callable.

    Raises:
        TypeError: If positional arguments are supplied for a mathematical constant.
        ValueError: If `name` does not identify a public :mod:`math` attribute.
    """
    value = getattr(math, name, None) if not name.startswith("_") else None
    if value is None:
        raise ValueError(f"Unknown public math attribute `{name}`.")
    if callable(value):
        return value(*args)
    if args:
        raise TypeError(f"Math constant `{name}` does not accept arguments.")
    return value


def import_from_module(dotpath: str) -> Any:
    """Import an object from its fully qualified dotted path.

    Args:
        dotpath: Module path followed by the requested attribute name.

    Returns:
        Imported module attribute.

    Raises:
        ValueError: If `dotpath` does not contain both a module and attribute name.
    """
    if "." not in dotpath:
        raise ValueError("An import path must contain a module and attribute name.")
    module_name, attribute_name = dotpath.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), attribute_name)


__all__ = ["import_from_module", "register_omegaconf_resolvers"]
