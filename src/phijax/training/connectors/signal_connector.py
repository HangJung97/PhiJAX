from __future__ import annotations

import logging
import signal
import threading
from types import FrameType
from typing import Any

log = logging.getLogger(__name__)


class _SignalConnector:
    """Convert process interruption signals into Trainer-owned graceful shutdown state.

    The first `SIGINT` or `SIGTERM` only records a stop request. The fit loop calls :meth:`raise_if_requested` at step
    boundaries, so interruption handling always sees a completed step. A repeated signal escalates immediately.
    """

    def __init__(self) -> None:
        """Initialize inactive signal handling and interruption state."""
        self.interrupted = False
        self.received_sigterm = False
        self._previous: dict[int, Any] = {}
        self._received = 0
        self._requested: int | None = None

    def reset(self) -> None:
        """Reset interruption flags before a fit call."""
        self.interrupted = False
        self.received_sigterm = False
        self._received = 0
        self._requested = None

    def install(self) -> None:
        """Install `SIGINT` and `SIGTERM` handlers on the main Python thread."""
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM):
            self._previous[signum] = signal.getsignal(signum)
            signal.signal(signum, self._handle_signal)

    def restore(self) -> None:
        """Restore every process signal handler replaced by :meth:`install`."""
        for signum, handler in self._previous.items():
            signal.signal(signum, handler)
        self._previous.clear()

    def raise_if_requested(self) -> None:
        """Raise the interruption for a pending stop request.

        Raises:
            KeyboardInterrupt: If the first signal was `SIGINT`.
            SystemExit: With status `143` if the first signal was `SIGTERM`.
        """
        if self._requested is None:
            return
        if self._requested == signal.SIGTERM:
            raise SystemExit(128 + self._requested)
        raise KeyboardInterrupt(f"Received {signal.Signals(self._requested).name}.")

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Record a stop request on the first signal and escalate repeated signals.

        Args:
            signum: Operating-system signal number.
            frame: Current interpreter frame supplied by :mod:`signal`.

        Raises:
            SystemExit: When a repeated signal cannot use its previous handler.
        """
        self._received += 1
        if signum == signal.SIGTERM:
            self.received_sigterm = True
        if self._received == 1:
            self._requested = signum
            log.warning(
                f"Received {signal.Signals(signum).name}; stopping after the current step. "
                "Send the signal again to stop immediately."
            )
            return

        previous = self._previous.get(signum, signal.SIG_DFL)
        signal.signal(signum, previous)
        if callable(previous):
            previous(signum, frame)
            return
        raise SystemExit(128 + signum)
