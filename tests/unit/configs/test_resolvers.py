import math

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import InterpolationResolutionError

from phijax.integrations.omegaconf import import_from_module, register_omegaconf_resolvers


def test_register_omegaconf_resolvers_is_idempotent_and_resolves_general_expressions() -> None:
    """Verify repeated registration and every general-purpose resolver with nested interpolation."""
    register_omegaconf_resolvers()
    register_omegaconf_resolvers()
    config = OmegaConf.create(
        {
            "pi": "${math:pi}",
            "square_root": "${math:sqrt,9}",
            "viscosity": "${op:truediv,0.01,${math:pi}}",
            "choice": "${op.ternary:${op:eq,2,2},selected,rejected}",
            "filename": "${call:os.path.basename,/tmp/sample.mat}",
            "items": "${tuple:a,b,3}",
            "valid": "${assert:${op:gt,3,2}}",
        }
    )

    assert config.pi == pytest.approx(math.pi)
    assert config.square_root == 3.0
    assert config.viscosity == pytest.approx(0.01 / math.pi)
    assert config.choice == "selected"
    assert config.filename == "sample.mat"
    assert config["items"] == ("a", "b", 3)
    assert config.valid is True


def test_assert_resolver_can_raise_or_warn(caplog: pytest.LogCaptureFixture) -> None:
    """Verify failed configuration assertions support strict and warning-only policies.

    Args:
        caplog: Captured Python logging records.
    """
    strict = OmegaConf.create({"valid": "${assert:false}"})
    warning_only = OmegaConf.create({"valid": "${assert:false,false}"})

    with pytest.raises(InterpolationResolutionError, match="Hydra configuration assertion failed"):
        _ = strict.valid
    assert warning_only.valid is False
    assert "Hydra configuration assertion failed" in caplog.text


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("${math:unknown}", "Unknown public math attribute"),
        ("${math:pi,2}", "does not accept arguments"),
        ("${op:unknown,1}", "Unknown public operator"),
        ("${call:math.pi}", "is not callable"),
    ],
)
def test_resolvers_reject_unknown_or_incompatible_targets(expression: str, message: str) -> None:
    """Verify dynamic resolver targets fail with contextual errors.

    Args:
        expression: Invalid interpolation under test.
        message: Expected wrapped resolution-error detail.
    """
    config = OmegaConf.create({"value": expression})
    with pytest.raises(InterpolationResolutionError, match=message):
        _ = config.value


def test_import_from_module_validates_and_imports_dotted_paths() -> None:
    """Verify dotted imports return requested objects and reject incomplete paths."""
    assert import_from_module("math.pi") == math.pi
    with pytest.raises(ValueError, match="module and attribute"):
        import_from_module("pi")
