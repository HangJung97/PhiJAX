from collections.abc import Sequence


def as_tuple(shape: int | Sequence[int]) -> tuple[int, ...]:
    """Normalize a scalar or integer sequence shape to a tuple.

    Args:
        shape: Scalar dimension or sequence of dimensions.

    Returns:
        Tuple containing all dimensions from `shape`.
    """
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)
