from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Callable
from uuid import uuid4


@dataclass(slots=True)
class TTSQueueEntry:
    """One immutable-input TTS request waiting for or using the single model worker."""

    task_id: str
    label: str
    output_path: Path
    generate: Callable[[], object] = field(repr=False, compare=False)
    cancel_event: Event = field(default_factory=Event, repr=False, compare=False)
    status: str = "Na fila"


class TTSJobQueue:
    """Small in-memory FIFO for desktop TTS tasks; it does not depend on Qt."""

    TERMINAL_STATES = frozenset({"Concluído", "Falhou", "Cancelado"})

    def __init__(self, max_history: int = 100) -> None:
        self._entries: list[TTSQueueEntry] = []
        self._current: TTSQueueEntry | None = None
        self.max_history = max(1, int(max_history))

    @property
    def current(self) -> TTSQueueEntry | None:
        return self._current

    @property
    def entries(self) -> tuple[TTSQueueEntry, ...]:
        return tuple(self._entries)

    @property
    def pending_count(self) -> int:
        return sum(entry.status == "Na fila" for entry in self._entries)

    @property
    def active_count(self) -> int:
        return self.pending_count + (1 if self._current is not None else 0)

    def enqueue(
        self,
        label: str,
        output_path: str | Path,
        generate: Callable[[], object],
    ) -> TTSQueueEntry:
        self._trim_history()
        entry = TTSQueueEntry(
            task_id=uuid4().hex[:12],
            label=str(label).strip() or "Texto sem título",
            output_path=Path(output_path).expanduser(),
            generate=generate,
        )
        self._entries.append(entry)
        return entry

    def start_next(self) -> TTSQueueEntry | None:
        if self._current is not None:
            return None
        for entry in self._entries:
            if entry.status == "Na fila":
                entry.status = "Gerando"
                self._current = entry
                return entry
        return None

    def request_cancel_current(self) -> bool:
        if self._current is None:
            return False
        if self._current.status == "Cancelamento solicitado":
            return False
        self._current.status = "Cancelamento solicitado"
        self._current.cancel_event.set()
        return True

    def finish_current(self, task_id: str, status: str) -> bool:
        if self._current is None or self._current.task_id != task_id:
            return False
        if status not in self.TERMINAL_STATES:
            raise ValueError(f"Estado final inválido: {status}")
        self._current.status = status
        self._current = None
        return True

    def remove(self, task_id: str) -> bool:
        """Remove queued or finished entries; the running item must be cancelled."""
        if self._current is not None and self._current.task_id == task_id:
            return False
        for index, entry in enumerate(self._entries):
            if entry.task_id == task_id:
                if entry.status in {"Gerando", "Cancelamento solicitado"}:
                    return False
                self._entries.pop(index)
                return True
        return False

    def clear_pending(self) -> int:
        """Discard only tasks that have not started; return the number removed."""
        before = len(self._entries)
        self._entries = [entry for entry in self._entries if entry.status != "Na fila"]
        return before - len(self._entries)

    def _trim_history(self) -> None:
        while len(self._entries) >= self.max_history:
            removable = next(
                (
                    index
                    for index, entry in enumerate(self._entries)
                    if entry.status in self.TERMINAL_STATES
                ),
                None,
            )
            if removable is None:
                return
            self._entries.pop(removable)
