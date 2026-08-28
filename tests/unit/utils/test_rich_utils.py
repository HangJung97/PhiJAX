from pathlib import Path

import pytest
from omegaconf import OmegaConf

from phijax.utils.rich_utils import print_config_tree


def test_print_config_tree_renders_and_saves_plain_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Verify Rich renders the composed tree and exports it below the run directory."""
    config = OmegaConf.create(
        {
            "application": {"name": "example"},
            "model": {"hidden": [4, 4]},
            "paths": {"output_dir": str(tmp_path)},
            "seed": 7,
        }
    )
    print_config_tree(config, resolve=True, save_to_file=True)
    captured = capsys.readouterr()
    saved = (tmp_path / "config_tree.log").read_text(encoding="utf-8")
    assert "CONFIG" in captured.out
    assert "model" in saved
    assert "hidden" in saved
    assert saved.index("application") < saved.index("model")


def test_print_config_tree_requires_output_path_when_saving() -> None:
    """Verify saving fails clearly when no Hydra output directory is configured."""
    config = OmegaConf.create({"model": {"hidden": [4]}})
    with pytest.raises(ValueError, match=r"cfg\.paths\.output_dir"):
        print_config_tree(config, save_to_file=True)


def test_print_config_tree_renders_missing_mandatory_values_without_resolving_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Verify config diagnostics render `???` instead of masking a later task-specific validation error.

    Args:
        capsys: Pytest standard-output capture fixture.
    """
    config = OmegaConf.create({"ckpt_path": "???"})
    print_config_tree(config, resolve=True)
    assert "???" in capsys.readouterr().out
