from collections.abc import Mapping, Sequence
from typing import Any


def pad_keys(
    mapping: Mapping[str, Any],
    prefix: str | None = None,
    postfix: str | None = None,
    exclude: str | Sequence[str] | None = None,
) -> dict[str, Any]:
    """Add a prefix and postfix to selected mapping keys.

    Args:
        mapping: Mapping whose string keys will be transformed.
        prefix: Text to prepend to keys, or `None` for no prefix.
        postfix: Text to append to keys, or `None` for no postfix.
        exclude: One key or a sequence of keys to leave unchanged.

    Returns:
        New dictionary containing the transformed keys and original values.
    """
    excluded = {exclude} if isinstance(exclude, str) else set(exclude or ())
    prefix = prefix or ""
    postfix = postfix or ""
    return {key if key in excluded else f"{prefix}{key}{postfix}": value for key, value in mapping.items()}
