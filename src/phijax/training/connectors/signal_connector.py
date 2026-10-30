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

    In a multi-process run, a signal stays pending until the fit loop combines it across processes and calls
    :meth:`request`, so every process stops at the same step even when only one process received the signal.
    """

    def __init__(self) -> None:
        """Initialize inactive signal handling and interruption state."""
        self.interrupted = False
        self.received_sigterm = False
        self._previous: dict[int, Any] = {}
        self._received = 0
        self._requested: int | None = None
        self._pending: int | None = None
        self._synchronized = False

    @property
    def pending_signal(self) -> int | None:
        """Return the first signal received in a multi-process run and not yet agreed by every process.

        Returns:
            Signal number, or `None` when no signal is pending.
        """
        return self._pending

    def request(self, signum: int) -> None:
        """Record a stop request agreed by every process.

        Args:
            signum: `SIGINT` or `SIGTERM`.
        """
        self._requested = signum
        if signum == signal.SIGTERM:
            self.received_sigterm = True

    def reset(self) -> None:
        """Reset interruption flags before a fit call."""
        self.interrupted = False
        self.received_sigterm = False
        self._received = 0
        self._requested = None
        self._pending = None

    def install(self, *, synchronized: bool = False, handle_sigterm: bool = True) -> None:
        """Install `SIGINT` and optionally `SIGTERM` handlers on the main Python thread.

        Args:
            synchronized: Whether signals stay pending until every process agrees, as in a multi-process run.
            handle_sigterm: Whether to handle `SIGTERM`. A Python handler would hide `SIGTERM` from JAX's preemption
                service, so multi-process runs leave it to JAX when that service is enabled.
        """
        self._synchronized = synchronized
        if threading.current_thread() is not threading.main_thread():
            return
        for signum in (signal.SIGINT, signal.SIGTERM) if handle_sigterm else (signal.SIGINT,):
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
        if self._received == 1:
            if self._synchronized:
                self._pending = signum
            else:
                self.request(signum)
            log.warning(
                f"Received {signal.Signals(signum).name}; stopping after the current step. "
                "Send the signal again to stop immediately."
            )
            return
        if signum == signal.SIGTERM:
            self.received_sigterm = True

        previous = self._previous.get(signum, signal.SIG_DFL)
        signal.signal(signum, previous)
        if callable(previous):
            previous(signum, frame)
            return
        raise SystemExit(128 + signum)
