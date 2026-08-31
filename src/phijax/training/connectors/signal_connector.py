from __future__ import annotations

import signal
import threading
from types import FrameType
from typing import Any


class _SignalConnector:
    """Convert process interruption signals into Trainer-owned graceful shutdown state."""

    def __init__(self) -> None:
        """Initialize inactive signal handling and interruption state."""
        self.interrupted = False
        self.received_sigterm = False
        self._previous: dict[int, Any] = {}
        self._received = 0

    def reset(self) -> None:
        """Reset interruption flags before a fit call."""
        self.interrupted = False
        self.received_sigterm = False
        self._received = 0

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

    def _handle_signal(self, signum: int, frame: FrameType | None) -> None:
        """Raise an interruption on the first signal and escalate repeated signals.

        Args:
            signum: Operating-system signal number.
            frame: Current interpreter frame supplied by :mod:`signal`.

        Raises:
            KeyboardInterrupt: On the first `SIGINT`.
            SystemExit: On the first `SIGTERM`, or when a repeated signal cannot use its previous handler.
        """
        self._received += 1
        if signum == signal.SIGTERM:
            self.received_sigterm = True
        if self._received == 1:
            if signum == signal.SIGTERM:
                raise SystemExit(128 + signum)
            signal_name = signal.Signals(signum).name
            raise KeyboardInterrupt(f"Received {signal_name}.")

        previous = self._previous.get(signum, signal.SIG_DFL)
        signal.signal(signum, previous)
        if callable(previous):
            previous(signum, frame)
            return
        raise SystemExit(128 + signum)
