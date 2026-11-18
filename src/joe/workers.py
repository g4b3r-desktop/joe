from __future__ import annotations

import time
import traceback
from collections.abc import Callable
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Signal

from joe.audio_chunks import merge_wav_files
from joe.audio_performance import AudioPerformance, wav_duration_seconds
from joe.task_control import CancellationToken


class WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class FunctionWorker(QRunnable):
    """Run a blocking task away from the GUI thread and discard cancelled results."""

    def __init__(self, function: Callable[[], object]) -> None:
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()
        self.cancel_token = CancellationToken()

    @property
    def cancel_requested(self) -> bool:
        return self.cancel_token.is_requested

    def cancel(self) -> None:
        """Request cancellation; a blocking model/API call is not forcibly interrupted."""
        self.cancel_token.request()

    def run(self) -> None:
        if self.cancel_requested:
            self.signals.cancelled.emit()
            return
        try:
            result = self.function()
        except Exception as exc:  # noqa: BLE001
            if self.cancel_requested:
                self.signals.cancelled.emit()
                return
            details = traceback.format_exc()
            self.signals.failed.emit(str(exc) + chr(10) * 2 + details)
            return

        if self.cancel_requested:
            self.signals.cancelled.emit()
        else:
            self.signals.finished.emit(result)


class TTSQueueWorkerSignals(QObject):
    progress = Signal(int, int, str, str)
    item_completed = Signal(object)
    item_failed = Signal(str, str)
    finished = Signal()
    cancelled = Signal()


class TTSQueueWorker(QRunnable):
    """Run queued TTS jobs sequentially; a failed item does not block the rest."""

    def __init__(self, items, generator: Callable[[dict], dict]) -> None:
        super().__init__()
        self.items = [dict(item) for item in items]
        self.generator = generator
        self.cancel_token = CancellationToken()
        self.signals = TTSQueueWorkerSignals()

    def cancel(self) -> None:
        self.cancel_token.request()

    def run(self) -> None:
        total = len(self.items)
        for index, item in enumerate(self.items, start=1):
            if self.cancel_token.is_requested:
                self.signals.cancelled.emit()
                return
            item_id = str(item.get("id", ""))
            title = str(item.get("title", "Tarefa"))
            self.signals.progress.emit(index, total, item_id, title)
            try:
                result = self.generator(item)
            except Exception as exc:  # noqa: BLE001
                self.signals.item_failed.emit(item_id, str(exc))
                continue
            result["item_id"] = item_id
            self.signals.item_completed.emit(result)

        if self.cancel_token.is_requested:
            self.signals.cancelled.emit()
        else:
            self.signals.finished.emit()


class ChunkWorkerSignals(QObject):
    chunk_ready = Signal(int, str, int)
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class ClipboardChunkWorker(QRunnable):
    """Generate speech chunks, measure desktop time and merge the final WAV."""

    def __init__(
        self,
        chunks,
        generator,
        output_directory: Path,
        final_path: Path,
        remote: bool = False,
    ):
        super().__init__()
        self.chunks = list(chunks)
        self.generator = generator
        self.output_directory = output_directory
        self.final_path = final_path
        self.remote = remote
        self.cancel_token = CancellationToken()
        self.signals = ChunkWorkerSignals()

    def cancel(self) -> None:
        self.cancel_token.request()

    def run(self) -> None:
        started_at = time.perf_counter()
        generated: list[Path] = []
        total = len(self.chunks)

        try:
            for index, text in enumerate(self.chunks, start=1):
                if self.cancel_token.is_requested:
                    self.signals.cancelled.emit()
                    return

                destination = self.output_directory / f"part_{index:03d}.wav"
                self.signals.progress.emit(f"Gerando trecho {index} de {total}…")
                result = self._generate_chunk(index, total, text, destination)

                if self.cancel_token.is_requested:
                    self.signals.cancelled.emit()
                    return

                generated.append(result)
                self.signals.chunk_ready.emit(index, str(result), total)

            if self.cancel_token.is_requested:
                self.signals.cancelled.emit()
                return

            self.signals.progress.emit("Unindo os trechos em um único WAV…")
            merged = merge_wav_files(generated, self.final_path)
            if self.cancel_token.is_requested:
                self.signals.cancelled.emit()
                return
            performance = AudioPerformance(
                elapsed_seconds=max(time.perf_counter() - started_at, 0.000001),
                audio_duration_seconds=wav_duration_seconds(merged),
                character_count=sum(len(chunk) for chunk in self.chunks),
                remote=self.remote,
            )
            self.signals.finished.emit(
                {"path": str(merged), "performance": performance}
            )
        except Exception as exc:  # noqa: BLE001
            if self.cancel_token.is_requested:
                self.signals.cancelled.emit()
            else:
                self.signals.failed.emit(str(exc))

    def _generate_chunk(
        self,
        index: int,
        total: int,
        text: str,
        destination: Path,
    ) -> Path:
        """Retry each chunk once before aborting the complete clipboard job."""
        last_error = None

        for attempt in range(2):
            if self.cancel_token.is_requested:
                raise RuntimeError("Geração cancelada.")
            try:
                if destination.exists():
                    destination.unlink()
                result = Path(self.generator(text, destination))
                if not result.is_file() or result.stat().st_size == 0:
                    raise RuntimeError("O serviço TTS não produziu um WAV válido.")
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if self.cancel_token.is_requested:
                    raise RuntimeError("Geração cancelada.") from exc
                if attempt == 0:
                    self.signals.progress.emit(
                        f"Trecho {index}: falha temporária; tentando novamente…"
                    )

        raise RuntimeError(
            f"Falha no trecho {index} de {total} após duas tentativas: {last_error}"
        ) from last_error
