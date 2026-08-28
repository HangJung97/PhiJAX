from collections.abc import Sequence
from pathlib import Path

from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.syntax import Syntax
from rich.tree import Tree

from phijax.utils.pylogger import get_colorlogger

log = get_colorlogger(__name__)


def print_config_tree(
    cfg: DictConfig,
    print_order: Sequence[str] = (
        "application",
        "data",
        "model",
        "callbacks",
        "logger",
        "trainer",
        "paths",
        "extras",
    ),
    resolve: bool = False,
    save_to_file: bool = False,
) -> None:
    """Print a composed Hydra configuration as a Rich tree.

    Args:
        cfg: Configuration composed by Hydra.
        print_order: Preferred order for top-level configuration fields. Present fields not listed here are appended in
            their configuration order.
        resolve: Whether to resolve interpolations before rendering configuration groups.
        save_to_file: Whether to save a plain-text rendering as `config_tree.log` in `cfg.paths.output_dir`.

    Raises:
        ValueError: If `save_to_file` is enabled but `cfg.paths.output_dir` is unavailable.
    """
    style = "dim"
    tree = Tree("CONFIG", style=style, guide_style=style)
    queue = [field for field in print_order if field in cfg]
    queue.extend(field for field in cfg if isinstance(field, str) and field not in queue)

    for field in queue:
        branch = tree.add(field, style=style, guide_style=style)
        if OmegaConf.is_missing(cfg, field):
            content = "???"
        else:
            config_group = cfg[field]
            content = (
                OmegaConf.to_yaml(config_group, resolve=resolve)
                if OmegaConf.is_config(config_group)
                else str(config_group)
            )
        branch.add(Syntax(content, "yaml", word_wrap=True))

    Console().print(tree)

    if save_to_file:
        output_dir = OmegaConf.select(cfg, "paths.output_dir")
        if output_dir is None:
            raise ValueError("`cfg.paths.output_dir` is required to save the Rich configuration tree.")
        output_path = Path(str(output_dir)) / "config_tree.log"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            Console(file=file, color_system=None, width=120).print(tree)
