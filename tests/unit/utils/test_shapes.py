import pytest

from phijax.utils.shapes import as_tuple


@pytest.mark.parametrize(("shape", "expected"), [(3, (3,)), ([2, 4], (2, 4)), ((1, 5), (1, 5))])
def test_as_tuple_normalizes_supported_shape_forms(
    shape: int | list[int] | tuple[int, ...], expected: tuple[int, ...]
) -> None:
    """Verify scalar, list, and tuple shapes share a stable tuple representation.

    Args:
        shape: Shape representation supplied by the parameterized test case.
        expected: Normalized tuple expected for `shape`.
    """
    assert as_tuple(shape) == expected
