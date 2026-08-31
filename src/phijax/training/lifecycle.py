from collections.abc import Iterable

from phijax.callbacks import Callback, CallbackContext
from phijax.core import BasePhiModule, PhiModuleContext
from phijax.training.loggers import ExperimentLogger


class TaskLifecycle:
    """Coordinate shared callback, module, logger, and teardown behavior for one Trainer task.

    Callback hooks are always dispatched before the matching module hook, following Lightning's default ordering.
    Numerical fit and prediction loops remain outside this helper so their control flow stays explicit and readable.
    """

    def __init__(
        self,
        callbacks: Iterable[Callback],
        module: BasePhiModule,
        logger: ExperimentLogger,
        *,
        is_global_zero: bool,
    ) -> None:
        """Initialize an inactive task lifecycle.

        Args:
            callbacks: Ordered host callbacks participating in the task.
            module: Application module receiving lifecycle hooks.
            logger: Experiment logger finalized when the task terminates.
            is_global_zero: Whether logger finalization belongs to this process.
        """
        self._callbacks = tuple(callbacks)
        self._module = module
        self._logger = logger
        self._is_global_zero = is_global_zero
        self._setup_callbacks: list[Callback] = []
        self._module_setup = False
        self._finalized = False

    @property
    def callbacks(self) -> tuple[Callback, ...]:
        """Return callbacks whose setup completed successfully.

        Returns:
            Ordered setup callback tuple.
        """
        return tuple(self._setup_callbacks)

    def setup(self) -> None:
        """Set up callbacks in declaration order and then set up the module."""
        if self._is_global_zero:
            self._logger.setup()
        for callback in self._callbacks:
            callback.setup()
            self._setup_callbacks.append(callback)
        self._module.setup()
        self._module_setup = True

    def handle_exception(
        self,
        exception: BaseException,
        callback_context: CallbackContext,
        module_context: PhiModuleContext,
    ) -> None:
        """Dispatch an exception to resources whose setup completed.

        Args:
            exception: Primary task exception.
            callback_context: Most recent valid callback context.
            module_context: Most recent valid module context.
        """
        for callback in self._setup_callbacks:
            callback.on_exception(exception, callback_context)
        if self._module_setup:
            self._module.on_exception(exception, module_context)

    def finalize(self, status: str) -> None:
        """Finalize the experiment logger at most once.

        Args:
            status: Terminal status such as `success`, `interrupted`, or `failed`.
        """
        if self._finalized or not self._is_global_zero:
            return
        self._finalized = True
        self._logger.finalize(status)

    def teardown(self) -> None:
        """Release setup callbacks followed by the application module."""
        for callback in self._setup_callbacks:
            callback.teardown()
        if self._module_setup:
            self._module.teardown()


__all__ = ["TaskLifecycle"]
