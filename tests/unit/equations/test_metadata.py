from functools import partial

import pytest

from phijax.equations import get_residual_names, residual_equation


def test_residual_equation_names_survive_configured_partials() -> None:
    """Verify static equation names can be discovered through nested configured partials."""

    @residual_equation(names=("continuity", "momentum"))
    def equation(value: int, *, scale: int = 1) -> int:
        """Return a scaled placeholder value.

        Args:
            value: Placeholder equation input.
            scale: Multiplicative placeholder coefficient.

        Returns:
            Scaled placeholder value.
        """
        return scale * value

    configured = partial(partial(equation, scale=2))

    assert get_residual_names(configured) == ("continuity", "momentum")
    assert configured(3) == 6


@pytest.mark.parametrize(
    ("names", "error_type", "message"),
    [
        ((), ValueError, "non-empty"),
        (("same", "same"), ValueError, "unique"),
        (("pde/continuity",), ValueError, "local names"),
        ((1,), TypeError, "strings"),
    ],
)
def test_residual_equation_validates_local_names(
    names: tuple[object, ...],
    error_type: type[Exception],
    message: str,
) -> None:
    """Verify invalid equation-local residual names fail during decoration setup.

    Args:
        names: Candidate local names.
        error_type: Expected validation exception type.
        message: Expected exception-message fragment.
    """
    with pytest.raises(error_type, match=message):
        residual_equation(names=names)  # type: ignore[arg-type]


def test_get_residual_names_requires_equation_metadata() -> None:
    """Verify name discovery gives an actionable error for an undecorated callable."""

    def equation(value: int) -> int:
        """Return a placeholder value.

        Args:
            value: Placeholder equation input.

        Returns:
            Unchanged placeholder value.
        """
        return value

    with pytest.raises(ValueError, match="configure `names` explicitly"):
        get_residual_names(equation)
