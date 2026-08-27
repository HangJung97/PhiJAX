import logging
from collections.abc import Mapping
from typing import Any

import colorlog


class RankedLogger(logging.LoggerAdapter):
    """Standard-library logger adapter with optional rank-zero-only emission.

    The adapter delegates to the logger registered under `name`, so handlers installed by Hydra remain responsible for
    formatting and routing records. When `rank_zero_only` is enabled, records created on nonzero JAX processes are
    discarded before they reach those handlers.
    """

    def __init__(
        self,
        name: str = __name__,
        *,
        rank_zero_only: bool = False,
        extra: Mapping[str, object] | None = None,
    ) -> None:
        """Initialize a JAX-aware logger adapter.

        Args:
            name: Name of the underlying standard-library logger.
            rank_zero_only: Whether to emit records only from JAX process zero.
            extra: Optional contextual values added to each emitted log record.
        """
        super().__init__(logging.getLogger(name), extra)
        self.rank_zero_only = rank_zero_only

    def log(self, level: int, msg: object, *args: object, **kwargs: Any) -> None:
        """Emit a record when its level and configured process policy allow it.

        Args:
            level: Standard-library logging level for the record.
            msg: Message object passed to the underlying logger.
            *args: Positional arguments forwarded to :meth:`logging.Logger.log`.
            **kwargs: Keyword arguments forwarded to :meth:`logging.Logger.log`.
        """
        if not self.isEnabledFor(level):
            return
        if self.rank_zero_only and _process_index() != 0:
            return
        kwargs["stacklevel"] = int(kwargs.get("stacklevel", 1)) + 1
        super().log(level, msg, *args, **kwargs)


def _process_index() -> int:
    """Return the current JAX process index without importing JAX during logger construction.

    Returns:
        Integer process index reported by JAX's distributed runtime.
    """
    import jax

    return int(jax.process_index())


def get_colorlogger(name: str = "app", level: int = logging.INFO) -> logging.Logger:
    """Create or retrieve a colorized Hydra-style logger.

    The logger owns one colorized stream handler and disables propagation so Hydra's root handler does not emit each
    message a second time. Repeated calls with the same name reuse the configured handler.

    Args:
        name: Logger name displayed in each record.
        level: Minimum standard-library logging level to emit.

    Returns:
        Configured :class:`logging.Logger` instance.
    """
    log_colors = {
        "DEBUG": "purple",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "red",
    }
    formatter = colorlog.ColoredFormatter(
        "[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s][%(log_color)s%(levelname)s%(reset)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors=log_colors,
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False
    if not any(getattr(handler, "_phijax_colorlog", False) for handler in logger.handlers):
        handler = colorlog.StreamHandler()
        handler.setFormatter(formatter)
        handler._phijax_colorlog = True  # type: ignore[attr-defined]
        logger.addHandler(handler)
    return logger
