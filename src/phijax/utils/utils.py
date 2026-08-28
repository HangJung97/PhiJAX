import warnings
from collections.abc import Callable
from contextvars import ContextVar
from functools import wraps
from importlib import import_module
from importlib.util import find_spec

from omegaconf import DictConfig, OmegaConf

from phijax.integrations.omegaconf import register_omegaconf_resolvers
from phijax.utils.pylogger import get_colorlogger
from phijax.utils.rich_utils import print_config_tree

log = get_colorlogger(__name__)

type TaskFinalizer = Callable[[str], None]

_task_finalizers: ContextVar[list[TaskFinalizer] | None] = ContextVar("phijax_task_finalizers", default=None)


def register_task_finalizer(finalizer: TaskFinalizer) -> bool:
    """Register an idempotent resource finalizer with the active task wrapper.

    Args:
        finalizer: Callable accepting the terminal status `success`, `interrupted`, or `failed`.

    Returns:
        Whether an active :func:`task_wrapper` accepted the finalizer.
    """
    finalizers = _task_finalizers.get()
    if finalizers is None:
        return False
    finalizers.append(finalizer)
    return True


def _finish_active_wandb(status: str) -> None:
    """Finish a globally active W&B run when the optional SDK is installed.

    Args:
        status: Terminal task status mapped to the W&B exit code.
    """
    if find_spec("wandb") is None:
        return
    wandb = import_module("wandb")
    if getattr(wandb, "run", None) is None:
        return
    log.info("Closing active Weights & Biases run.")
    finish = getattr(wandb, "finish", None)
    if finish is not None:
        finish(exit_code=0 if status == "success" else 1)
    else:
        wandb.run.finish(exit_code=0 if status == "success" else 1)


def pre_hydra_routine() -> None:
    """Configure repository paths and environment variables before Hydra starts.

    The repository root is discovered from this source file using `pyproject.toml`, exported as `PROJECT_ROOT`, and used
    to load an optional repository-level `.env` file. PhiJAX's process-global OmegaConf resolvers are registered before
    Hydra composes a config. The working directory and Python import path are unchanged.
    """
    import rootutils

    rootutils.setup_root(
        __file__,
        indicator="pyproject.toml",
        project_root_env_var=True,
        dotenv=True,
        pythonpath=False,
        cwd=False,
    )
    register_omegaconf_resolvers()


def extras(cfg: DictConfig) -> None:
    """Apply optional warning and Rich-rendering utilities before a task starts.

    Args:
        cfg: Configuration containing an optional `extras` group and `paths.output_dir`.
    """
    extras_cfg = cfg.get("extras")
    if extras_cfg is None:
        log.warning("Extras config not found; optional entrypoint utilities are disabled.")
        return

    if extras_cfg.get("ignore_warnings", False):
        log.info("Disabling Python warnings.")
        warnings.filterwarnings("ignore")
    if extras_cfg.get("print_config", False):
        log.info("Printing the composed configuration with Rich.")
        print_config_tree(cfg, resolve=True, save_to_file=True)


def task_wrapper[TaskReturn](task_func: Callable[[DictConfig], TaskReturn]) -> Callable[[DictConfig], TaskReturn]:
    """Decorate a Hydra task with assertion resolution and consistent lifecycle logging.

    Lifecycle owners finalize stage-local resources normally. Longer-lived resources with idempotent cleanup, such as
    trainer checkpoint services and remote experiment loggers, may register a failure-safe finalizer through
    :func:`register_task_finalizer`; the wrapper invokes those finalizers in last-created-first-closed order.

    Args:
        task_func: Task callable accepting the composed configuration as its only argument.

    Returns:
        Wrapped callable that preserves the task's return value and re-raises failures with their original traceback.
    """

    @wraps(task_func)
    def wrap(cfg: DictConfig) -> TaskReturn:
        token = _task_finalizers.set([])
        status = "success"
        task_failed = False
        try:
            if assertions := cfg.get("_assert_"):
                log.info("Resolving configuration assertions.")
                OmegaConf.resolve(assertions)
            return task_func(cfg)
        except SystemExit:
            status = "interrupted"
            task_failed = True
            log.warning("Task interrupted.")
            raise
        except BaseException:
            status = "failed"
            task_failed = True
            log.exception("Task execution failed.")
            raise
        finally:
            cleanup_error: Exception | None = None
            try:
                output_dir = OmegaConf.select(cfg, "paths.output_dir", default=None)
                if output_dir is not None:
                    log.info(f"Output directory: {output_dir}")
            except Exception as error:
                log.exception("Failed to resolve the task output directory.")
                cleanup_error = error
            finalizers = tuple(reversed(_task_finalizers.get() or ()))
            for finalizer in finalizers:
                try:
                    finalizer(status)
                except Exception as error:
                    log.exception("Task resource finalization failed.")
                    if cleanup_error is None:
                        cleanup_error = error
            try:
                _finish_active_wandb(status)
            except Exception as error:
                log.exception("Failed to finish the active Weights & Biases run.")
                if cleanup_error is None:
                    cleanup_error = error
            _task_finalizers.reset(token)
            if cleanup_error is not None and not task_failed:
                raise cleanup_error

    return wrap
