from collections.abc import Callable
from contextvars import ContextVar

type TaskFinalizer = Callable[[str], None]

# `task_wrapper` sets a fresh list for each task; outside a task the registry is `None`.
_task_finalizers: ContextVar[list[TaskFinalizer] | None] = ContextVar("phijax_task_finalizers", default=None)


def register_task_finalizer(finalizer: TaskFinalizer) -> bool:
    """Register an idempotent resource finalizer with the active task wrapper.

    Args:
        finalizer: Callable accepting the terminal status `success`, `interrupted`, or `failed`.

    Returns:
        Whether an active :func:`phijax.utils.task_wrapper` accepted the finalizer.
    """
    finalizers = _task_finalizers.get()
    if finalizers is None:
        return False
    finalizers.append(finalizer)
    return True
