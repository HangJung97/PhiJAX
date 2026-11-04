from typing import TYPE_CHECKING

from phijax._lazy import lazy_exports

if TYPE_CHECKING:
    from phijax.utils.format import as_numpy, to_plain_container
    from phijax.utils.logging_utils import pad_keys
    from phijax.utils.pylogger import RankedLogger, get_colorlogger
    from phijax.utils.rich_utils import print_config_tree
    from phijax.utils.seeding import resolve_seed, seed_everything
    from phijax.utils.shapes import as_tuple
    from phijax.utils.utils import extras, pre_hydra_routine, register_task_finalizer, task_wrapper

__getattr__, __dir__ = lazy_exports(
    __name__,
    {
        "RankedLogger": "phijax.utils.pylogger",
        "as_numpy": "phijax.utils.format",
        "as_tuple": "phijax.utils.shapes",
        "extras": "phijax.utils.utils",
        "get_colorlogger": "phijax.utils.pylogger",
        "pad_keys": "phijax.utils.logging_utils",
        "pre_hydra_routine": "phijax.utils.utils",
        "print_config_tree": "phijax.utils.rich_utils",
        "register_task_finalizer": "phijax.utils.utils",
        "resolve_seed": "phijax.utils.seeding",
        "seed_everything": "phijax.utils.seeding",
        "task_wrapper": "phijax.utils.utils",
        "to_plain_container": "phijax.utils.format",
    },
)

__all__ = [
    "RankedLogger",
    "as_numpy",
    "as_tuple",
    "extras",
    "get_colorlogger",
    "pad_keys",
    "pre_hydra_routine",
    "print_config_tree",
    "register_task_finalizer",
    "resolve_seed",
    "seed_everything",
    "task_wrapper",
    "to_plain_container",
]
