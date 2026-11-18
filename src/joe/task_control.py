from __future__ import annotations

from threading import Event


class CancellationToken:
    """Thread-safe cooperative cancellation flag."""

    def __init__(self) -> None:
        self._event = Event()

    @property
    def is_requested(self) -> bool:
        return self._event.is_set()

    def request(self) -> None:
        self._event.set()
