import sys
from collections.abc import Callable, Mapping
from importlib import import_module
from typing import Any


def lazy_exports(package: str, exports: Mapping[str, str]) -> tuple[Callable[[str], Any], Callable[[], list[str]]]:
    """Create PEP 562 hooks that import a package's public names on first access.

    Args:
        package: Name of the package whose `__init__` installs the hooks.
        exports: Defining module for every public name. A `module:attribute` value exports `attribute` under a
            different public name.

    Returns:
        Module-level `__getattr__` and `__dir__` functions for `package`.
    """

    def get_attribute(name: str) -> Any:
        """Import a public name from its defining module and cache it on the package.

        Args:
            name: Requested attribute name.

        Returns:
            The exported object.

        Raises:
            AttributeError: If `name` is not a public export of `package`.
        """
        target = exports.get(name)
        if target is None:
            raise AttributeError(f"module {package!r} has no attribute {name!r}")
        module_name, _, attribute = target.partition(":")
        value = getattr(import_module(module_name), attribute or name)
        # Later lookups find the cached value directly and skip this hook.
        setattr(sys.modules[package], name, value)
        return value

    def list_attributes() -> list[str]:
        """List loaded attributes together with every public export.

        Returns:
            Sorted attribute names of `package`.
        """
        return sorted({*vars(sys.modules[package]), *exports})

    return get_attribute, list_attributes
