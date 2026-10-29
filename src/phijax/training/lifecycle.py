import logging
from collections.abc import Callable, Iterable
from functools import partial

from phijax.callbacks import Callback, CallbackContext
from phijax.core import BasePhiModule, PhiModuleContext
from phijax.training.loggers import ExperimentLogger

log = logging.getLogger(__name__)


class TaskLifecycle:
    """Coordinate shared callback, module, logger, and teardown behavior for one Trainer task.

    Callback hooks are always dispatched before the matching module hook, following Lightning's default ordering.
    Numerical fit and prediction loops remain outside this helper so their control flow stays explicit and readable.

    Cleanup is exception-safe: after a task error, every `on_exception` hook, logger finalization, and teardown hook
    still runs. A failing cleanup hook is logged and attached to the task error as a note instead of replacing it.
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
        self._error: BaseException | None = None

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
        """Dispatch an exception to every resource whose setup completed.

        Args:
            exception: Primary task exception. Later cleanup failures are attached to it as notes.
            callback_context: Most recent valid callback context.
            module_context: Most recent valid module context.
        """
        self._error = exception
        for callback in self._setup_callbacks:
            self._cleanup(
                f"{type(callback).__name__}.on_exception",
                partial(callback.on_exception, exception, callback_context),
            )
        if self._module_setup:
            self._cleanup(
                f"{type(self._module).__name__}.on_exception",
                partial(self._module.on_exception, exception, module_context),
            )

    def finalize(self, status: str) -> None:
        """Finalize the experiment logger at most once.

        Args:
            status: Terminal status such as `success`, `interrupted`, or `failed`.
        """
        if self._finalized or not self._is_global_zero:
            return
        self._finalized = True
        if self._error is None:
            self._logger.finalize(status)
        else:
            self._cleanup(f"{type(self._logger).__name__}.finalize", partial(self._logger.finalize, status))

    def teardown(self) -> None:
        """Release every setup callback followed by the application module.

        Raises:
            Exception: The first teardown failure when the task itself succeeded. Later failures are attached as notes.
        """
        failures: list[Exception] = []
        hooks = [(type(callback).__name__, callback.teardown) for callback in self._setup_callbacks]
        if self._module_setup:
            hooks.append((type(self._module).__name__, self._module.teardown))
        for owner, hook in hooks:
            failure = self._cleanup(f"{owner}.teardown", hook)
            if failure is not None:
                failures.append(failure)
        if failures and self._error is None:
            for failure in failures[1:]:
                failures[0].add_note(f"Another teardown hook also failed: {failure!r}")
            raise failures[0]

    def _cleanup(self, description: str, hook: Callable[[], None]) -> Exception | None:
        """Run one cleanup hook without letting its failure stop the remaining cleanup.

        Args:
            description: Hook name used in log messages and notes.
            hook: Cleanup hook to run.

        Returns:
            The hook's exception, or `None` if it succeeded.
        """
        try:
            hook()
        except Exception as failure:
            # After a task error, the failure is logged and noted because only the task error propagates.
            if self._error is not None:
                log.error(f"{description} failed during task cleanup.", exc_info=failure)
                self._error.add_note(f"{description} also failed during cleanup: {failure!r}")
            return failure
        return None


__all__ = ["TaskLifecycle"]
