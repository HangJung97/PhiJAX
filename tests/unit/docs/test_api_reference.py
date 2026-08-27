import ast
import importlib
import re
from pathlib import Path
from typing import Any

import pytest

_ROOT = Path(__file__).parents[3]
_API_DIRECTORY = _ROOT / "docs" / "api"
_DIRECTIVE_PATTERN = re.compile(r"^::: (?P<path>[A-Za-z_][A-Za-z0-9_.]*)$", re.MULTILINE)


def _api_markdown() -> str:
    """Collect API page contents in stable path order.

    Returns:
        Concatenated API Markdown used for coverage assertions.
    """
    return "\n".join(path.read_text(encoding="utf-8") for path in sorted(_API_DIRECTORY.glob("*.md")))


def _resolve_documented_object(dotpath: str) -> Any:
    """Import one mkdocstrings target and traverse its attribute suffix.

    Args:
        dotpath: Fully qualified module, class, function, or attribute path.

    Returns:
        Imported object selected by `dotpath`.

    Raises:
        ImportError: If no prefix of `dotpath` identifies an importable module.
        AttributeError: If an imported module does not expose the documented attribute suffix.
    """
    parts = dotpath.split(".")
    for boundary in range(len(parts), 0, -1):
        module_name = ".".join(parts[:boundary])
        try:
            value = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            if error.name != module_name:
                raise
            continue
        for attribute in parts[boundary:]:
            value = getattr(value, attribute)
        return value
    raise ImportError(f"No importable module prefix exists for `{dotpath}`.")


def _exported_names(init_path: Path) -> tuple[str, ...]:
    """Read a static `__all__` declaration without importing its package.

    Args:
        init_path: Package initializer containing an optional literal `__all__` assignment.

    Returns:
        Exported names, or an empty tuple when the initializer declares no public export list.
    """
    tree = ast.parse(init_path.read_text(encoding="utf-8"))
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if any(isinstance(target, ast.Name) and target.id == "__all__" for target in node.targets):
            return tuple(ast.literal_eval(node.value))
    return ()


def documented_targets() -> tuple[str, ...]:
    """Return every unique mkdocstrings target from the API pages.

    Returns:
        Targets in page and declaration order with duplicates removed.
    """
    return tuple(dict.fromkeys(_DIRECTIVE_PATTERN.findall(_api_markdown())))


@pytest.mark.parametrize("dotpath", documented_targets())
def test_mkdocstrings_target_resolves(dotpath: str) -> None:
    """Verify every generated API block points to an importable Python object.

    Args:
        dotpath: Mkdocstrings target collected from the API pages.
    """
    assert _resolve_documented_object(dotpath) is not None


def test_every_package_export_appears_in_the_api_reference() -> None:
    """Verify public package exports remain discoverable in the split API reference."""
    documentation = _api_markdown()
    exported = {
        name for init_path in (_ROOT / "src" / "phijax").rglob("__init__.py") for name in _exported_names(init_path)
    }
    missing = sorted(name for name in exported if name not in documentation)

    assert not missing, f"Public exports missing from API documentation: {missing}"


def test_top_level_api_exposes_beta_runtime_contracts() -> None:
    """Verify the beta package root exposes its documented framework contracts."""
    import phijax

    expected = {
        "DataModule",
        "InitializedModel",
        "LossBalancer",
        "PhiModule",
        "TrainState",
        "Trainer",
        "TrainingPlan",
        "hessian_diagonal",
        "value_and_jacobian",
    }

    assert expected <= set(phijax.__all__)
    assert phijax.DataModule is phijax.PhiDataModule
