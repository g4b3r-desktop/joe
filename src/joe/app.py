from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import uuid
from collections.abc import Callable
from pathlib import Path

# Import PyTorch before Qt Multimedia; importing it later from a Qt worker can abort natively.
import torch  # noqa: F401

# isort: split
from PySide6.QtCore import (
    QObject, QRunnable, QSettings, QStandardPaths, QThreadPool,
    QTime, QTimer, QUrl, Qt, Signal,
)
from PySide6.QtGui import QAction, QDesktopServices, QIcon
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHeaderView,
    QInputDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from joe.audio_chunks import merge_wav_files, split_text_into_chunks
from joe.audio_history import AudioHistoryStore
from joe.generation_queue import GenerationQueue, GenerationTask
from joe.ui.audio_history import AudioHistoryPanel
from joe.ui.advanced_player import AdvancedAudioPlayer
from joe.audio_performance import (
    AudioPerformance,
    format_duration,
    format_elapsed_status,
    format_performance_report,
    wav_duration_seconds,
)
from joe.local_ipc import LocalCommandServer, send_local_command
from joe.playback import (
    decide_clipboard_shortcut_action,
    decide_playback_action,
)
from joe.pronunciation_dictionary import (
    load_pronunciation_dictionary_json,
    merge_pronunciation_entries,
)
from joe.services.colab_api import ColabAPIClient, ColabAPIError
from joe.services.parakeet import ParakeetReduxService
from joe.services.pocket_tts import PocketTTSService
from joe.text_normalization import normalize_text_for_tts
from joe.pronunciation_rules import (
    load_pronunciation_rules,
    serialize_pronunciation_rules,
)
from joe.voice_favorites import (
    canonical_voice_path,
    load_voice_favorites,
    serialize_voice_favorites,
)
from joe.voice_profiles import load_voice_profiles, serialize_voice_profiles
from joe.utils import build_srt, build_audio_output_path, write_bytes_atomically
from joe.transcript_sync import active_segment_index, normalize_timed_segments


class WorkerSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class FunctionWorker(QRunnable):
    """Run a blocking model call away from the GUI thread."""

    def __init__(self, function: Callable[[], object]) -> None:
        super().__init__()
        self.function = function
        self.signals = WorkerSignals()

    def run(self) -> None:
        try:
            self.signals.finished.emit(self.function())
        except Exception as exc:  # noqa: BLE001
            details = traceback.format_exc()
            self.signals.failed.emit(f"{exc}\n\n{details}")


class ChunkWorkerSignals(QObject):
    chunk_ready = Signal(int, str, int)
    progress = Signal(str)
    finished = Signal(object)
    failed = Signal(str)
    cancelled = Signal()


class ClipboardChunkWorker(QRunnable):
    """Gera blocos de fala, mede o tempo no desktop e une o WAV final."""

    def __init__(
        self,
        chunks,
        generator,
        output_directory: Path,
        final_path: Path,
        remote: bool = False,
    ):
        super().__init__()
        import threading

        self.chunks = list(chunks)
        self.generator = generator
        self.output_directory = output_directory
        self.final_path = final_path
        self.remote = remote
        self.cancel_event = threading.Event()
        self.signals = ChunkWorkerSignals()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        started_at = time.perf_counter()
        generated: list[Path] = []
        total = len(self.chunks)

        try:
            for index, text in enumerate(self.chunks, start=1):
                if self.cancel_event.is_set():
                    self.signals.cancelled.emit()
                    return

                destination = self.output_directory / f"part_{index:03d}.wav"
                self.signals.progress.emit(f"Gerando trecho {index} de {total}…")
                result = self._generate_chunk(index, total, text, destination)

                if self.cancel_event.is_set():
                    self.signals.cancelled.emit()
                    return

                generated.append(result)
                self.signals.chunk_ready.emit(index, str(result), total)

            if self.cancel_event.is_set():
                self.signals.cancelled.emit()
                return

            self.signals.progress.emit("Unindo os trechos em um único WAV…")
            merged = merge_wav_files(generated, self.final_path)
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
            self.signals.failed.emit(str(exc))

    def _generate_chunk(
        self,
        index: int,
        total: int,
        text: str,
        destination: Path,
    ) -> Path:
        """Tenta cada bloco duas vezes antes de interromper a tarefa."""
        last_error = None

        for attempt in range(2):
            try:
                if destination.exists():
                    destination.unlink()
                result = Path(self.generator(text, destination))
                if not result.is_file() or result.stat().st_size == 0:
                    raise RuntimeError("O serviço TTS não produziu um WAV válido.")
                return result
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                if attempt == 0:
                    self.signals.progress.emit(
                        f"Trecho {index}: falha temporária; tentando novamente…"
                    )

        raise RuntimeError(
            f"Falha no trecho {index} de {total} após duas tentativas: {last_error}"
        ) from last_error




class TTSGenerationSignals(QObject):
    progress = Signal(str, str)
    finished = Signal(str, object)
    failed = Signal(str, str)
    cancelled = Signal(str)


class TTSGenerationWorker(QRunnable):
    """Generate one queued TTS request in cancellable sentence-sized blocks."""

    def __init__(self, task: GenerationTask, tts_service: PocketTTSService) -> None:
        super().__init__()
        self.task = task
        self.tts_service = tts_service
        self.cancel_event = threading.Event()
        self.signals = TTSGenerationSignals()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        started_at = time.perf_counter()
        try:
            chunks = split_text_into_chunks(self.task.speech_text, max_chars=280)
            if not chunks:
                raise ValueError("O texto preparado para fala está vazio.")

            remote = (
                ColabAPIClient(self.task.remote_url, self.task.remote_token)
                if self.task.remote
                else None
            )
            with tempfile.TemporaryDirectory(prefix="joe-tts-queue-") as temp_dir:
                temporary_root = Path(temp_dir)
                generated: list[Path] = []

                for index, chunk_text in enumerate(chunks, start=1):
                    if self.cancel_event.is_set():
                        self.signals.cancelled.emit(self.task.task_id)
                        return

                    self.signals.progress.emit(
                        self.task.task_id,
                        f"Gerando trecho {index} de {len(chunks)}…",
                    )
                    chunk_path = temporary_root / f"part_{index:04d}.wav"
                    last_error: Exception | None = None

                    for attempt in range(2):
                        if self.cancel_event.is_set():
                            self.signals.cancelled.emit(self.task.task_id)
                            return
                        chunk_path.unlink(missing_ok=True)
                        try:
                            if remote is not None:
                                audio_data = remote.generate_audio(
                                    chunk_text,
                                    self.task.language,
                                    self.task.voice_path,
                                    cancel_event=self.cancel_event,
                                )
                                if self.cancel_event.is_set():
                                    self.signals.cancelled.emit(self.task.task_id)
                                    return
                                write_bytes_atomically(chunk_path, audio_data)
                            else:
                                self.tts_service.generate(
                                    chunk_text,
                                    self.task.language,
                                    chunk_path,
                                    self.task.voice_path,
                                )
                            if not chunk_path.is_file() or chunk_path.stat().st_size == 0:
                                raise RuntimeError("O serviço TTS não produziu um WAV válido.")
                            last_error = None
                            break
                        except ColabAPIError as exc:
                            if exc.cancelled or self.cancel_event.is_set():
                                self.signals.cancelled.emit(self.task.task_id)
                                return
                            last_error = exc
                        except Exception as exc:  # noqa: BLE001
                            if self.cancel_event.is_set():
                                self.signals.cancelled.emit(self.task.task_id)
                                return
                            last_error = exc

                        if attempt == 0:
                            self.signals.progress.emit(
                                self.task.task_id,
                                f"Trecho {index}: falha temporária; tentando novamente…",
                            )

                    if last_error is not None:
                        raise RuntimeError(
                            f"Falha no trecho {index} de {len(chunks)} após duas tentativas: "
                            f"{last_error}"
                        ) from last_error
                    generated.append(chunk_path)

                if self.cancel_event.is_set():
                    self.signals.cancelled.emit(self.task.task_id)
                    return

                self.signals.progress.emit(self.task.task_id, "Unindo os trechos em WAV…")
                merged_path = temporary_root / "complete.wav"
                merge_wav_files(generated, merged_path)

                if self.cancel_event.is_set():
                    self.signals.cancelled.emit(self.task.task_id)
                    return

                output_path = write_bytes_atomically(
                    self.task.output_path,
                    merged_path.read_bytes(),
                )
                performance = AudioPerformance(
                    elapsed_seconds=max(time.perf_counter() - started_at, 0.000001),
                    audio_duration_seconds=wav_duration_seconds(output_path),
                    character_count=len(self.task.speech_text),
                    remote=self.task.remote,
                )
                self.signals.finished.emit(
                    self.task.task_id,
                    {"path": str(output_path), "performance": performance},
                )
        except Exception as exc:  # noqa: BLE001
            if self.cancel_event.is_set():
                self.signals.cancelled.emit(self.task.task_id)
            else:
                self.signals.failed.emit(self.task.task_id, str(exc))


class GenerationQueueDialog(QDialog):
    """Compact control panel for queued TTS jobs; task text is never displayed."""

    def __init__(
        self,
        queue: GenerationQueue,
        add_callback: Callable[[], None],
        cancel_callback: Callable[[str], None],
        retry_callback: Callable[[str], None],
        prioritize_callback: Callable[[str], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.queue = queue
        self.add_callback = add_callback
        self.cancel_callback = cancel_callback
        self.retry_callback = retry_callback
        self.prioritize_callback = prioritize_callback
        self.setWindowTitle("Fila de geração")
        self.resize(720, 430)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "As tarefas são executadas em sequência. O texto fica somente na memória "
            "durante esta sessão e não é exibido nesta lista."
        ))
        self.task_list = QListWidget()
        layout.addWidget(self.task_list, 1)

        buttons = QHBoxLayout()
        self.add_button = QPushButton("Adicionar texto atual")
        self.cancel_button = QPushButton("Cancelar tarefa")
        self.retry_button = QPushButton("Repetir falha")
        self.priority_button = QPushButton("Prioridade urgente")
        self.clear_button = QPushButton("Limpar concluídas")
        self.close_button = QPushButton("Fechar")
        for button in (
            self.add_button,
            self.cancel_button,
            self.retry_button,
            self.priority_button,
            self.clear_button,
            self.close_button,
        ):
            buttons.addWidget(button)
        layout.addLayout(buttons)

        self.add_button.clicked.connect(lambda: self.add_callback())
        self.cancel_button.clicked.connect(self._cancel_selected)
        self.retry_button.clicked.connect(self._retry_selected)
        self.priority_button.clicked.connect(self._prioritize_selected)
        self.clear_button.clicked.connect(self._clear_finished)
        self.close_button.clicked.connect(self.close)

        self.refresh_timer = QTimer(self)
        self.refresh_timer.setInterval(700)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh_timer.start()
        self.refresh()

    def _selected_task_id(self) -> str | None:
        item = self.task_list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole)) if item else None

    def refresh(self) -> None:
        selected_id = self._selected_task_id()
        self.task_list.clear()
        labels = {
            "queued": "Na fila",
            "running": "Em execução",
            "cancelling": "Cancelando",
            "completed": "Concluída",
            "failed": "Falhou",
            "cancelled": "Cancelada",
        }
        selected_item = None
        for task in self.queue.tasks():
            prefix = "⚡ Urgente · " if task.urgent else ""
            label = labels.get(task.status, task.status)
            item = QListWidgetItem(
                f"{prefix}{task.output_name} — {label} — {task.progress}"
            )
            item.setData(Qt.ItemDataRole.UserRole, task.task_id)
            if task.error:
                item.setToolTip(task.error)
            self.task_list.addItem(item)
            if task.task_id == selected_id:
                selected_item = item
        if selected_item is not None:
            self.task_list.setCurrentItem(selected_item)

        task = self.queue.get(self._selected_task_id() or "")
        self.cancel_button.setEnabled(bool(task and task.status in {"queued", "running"}))
        self.retry_button.setEnabled(bool(task and task.status in {"failed", "cancelled"}))
        self.priority_button.setEnabled(bool(task and task.status == "queued" and not task.urgent))
        self.clear_button.setEnabled(
            any(item.status in self.queue.FINISHED_STATES for item in self.queue.tasks())
        )

    def _cancel_selected(self) -> None:
        task_id = self._selected_task_id()
        if task_id:
            self.cancel_callback(task_id)
        self.refresh()

    def _retry_selected(self) -> None:
        task_id = self._selected_task_id()
        if task_id:
            self.retry_callback(task_id)
        self.refresh()

    def _prioritize_selected(self) -> None:
        task_id = self._selected_task_id()
        if task_id:
            self.prioritize_callback(task_id)
        self.refresh()

    def _clear_finished(self) -> None:
        self.queue.remove_finished()
        self.refresh()

    def showEvent(self, event) -> None:  # noqa: N802
        self.refresh_timer.start()
        self.refresh()
        super().showEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802
        self.refresh_timer.stop()
        super().closeEvent(event)



class VoiceLibraryDialog(QDialog):
    """Recursive voice-file browser with audio preview."""

    AUDIO_SUFFIXES = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
    VOICE_SUFFIXES = AUDIO_SUFFIXES | {".safetensors"}

    def __init__(
        self, root: Path, player: QMediaPlayer,
        selected_path: Path | None = None, parent=None
    ):
        super().__init__(parent)
        self.root = root
        self.player = player
        self.initial_selected_path = selected_path
        self.selected_path: Path | None = None
        self.total_files = 0
        self.restore_warning = ""
        self.setWindowTitle("Biblioteca de vozes")
        self.resize(760, 560)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Pasta: {root}"))

        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Buscar por nome ou caminho…")
        self.search.textChanged.connect(self._filter_items)
        filters.addWidget(self.search, 1)

        self.type_filter = QComboBox()
        self.type_filter.addItem("Todos os formatos", "*")
        self.type_filter.addItem("WAV (*.wav)", ".wav")
        self.type_filter.addItem("MP3 (*.mp3)", ".mp3")
        self.type_filter.addItem("FLAC (*.flac)", ".flac")
        self.type_filter.addItem("M4A (*.m4a)", ".m4a")
        self.type_filter.addItem("OGG (*.ogg)", ".ogg")
        self.type_filter.addItem("AAC (*.aac)", ".aac")
        self.type_filter.addItem("Estado de voz (*.safetensors)", ".safetensors")
        self.type_filter.setMinimumWidth(190)
        self.type_filter.setToolTip("Mostrar somente arquivos do formato selecionado")
        self.type_filter.currentIndexChanged.connect(
            lambda: self._filter_items(self.search.text())
        )
        filters.addWidget(self.type_filter)
        layout.addLayout(filters)

        self.files = QListWidget()
        self.files.itemDoubleClicked.connect(lambda _item: self._accept_selection())
        self.files.setToolTip(
            "Clique em um áudio ou navegue com as setas ↑/↓ para ouvir. "
            "Dê duplo clique ou use “Usar esta voz” para selecionar a voz."
        )
        layout.addWidget(self.files, 1)
        self.status = QLabel("Procurando arquivos nas subpastas…")
        self.status.setObjectName("muted")
        layout.addWidget(self.status)
        actions = QHBoxLayout()
        stop = QPushButton("Parar reprodução")
        stop.clicked.connect(self.player.stop)
        actions.addWidget(stop)
        actions.addStretch(1)
        layout.addLayout(actions)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Usar esta voz")
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self._scan()
        # A conexão vem depois do scan para não tocar automaticamente a seleção restaurada.
        self.files.itemClicked.connect(self._preview_item)
        self.files.currentItemChanged.connect(self._preview_selection_changed)

    def _scan(self) -> None:
        found = []
        try:
            for path in self.root.rglob("*"):
                try:
                    if path.is_file() and path.suffix.lower() in self.VOICE_SUFFIXES:
                        found.append(path)
                except OSError:
                    continue
        except OSError as exc:
            self.status.setText(f"Não foi possível ler a pasta: {exc}")
            return
        for path in sorted(found, key=lambda item: str(item).casefold()):
            item = QListWidgetItem(str(path.relative_to(self.root)))
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(str(path))
            self.files.addItem(item)
        self.total_files = len(found)
        self._restore_selection()
        self._filter_items(self.search.text())

    def _restore_selection(self) -> None:
        if self.initial_selected_path is None:
            return
        try:
            target = self.initial_selected_path.expanduser().resolve()
        except OSError:
            target = self.initial_selected_path.expanduser()
        for index in range(self.files.count()):
            item = self.files.item(index)
            try:
                candidate = Path(item.data(Qt.ItemDataRole.UserRole)).resolve()
            except OSError:
                candidate = Path(item.data(Qt.ItemDataRole.UserRole))
            if candidate == target:
                self.files.setCurrentItem(item)
                self.files.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
                self.status.setText(
                    f"Voz selecionada anteriormente: {item.text()}"
                )
                return
        self.restore_warning = (
            "A voz selecionada anteriormente não está nesta pasta."
        )

    def _filter_items(self, query: str = "") -> None:
        """Combine the name/path search with the selected file extension."""
        query = query.casefold().strip()
        extension = str(self.type_filter.currentData())
        visible_count = 0

        for index in range(self.files.count()):
            item = self.files.item(index)
            raw_path = str(item.data(Qt.ItemDataRole.UserRole))
            full_path = raw_path.casefold()
            matches_query = (
                not query
                or query in item.text().casefold()
                or query in full_path
            )
            matches_type = (
                extension == "*"
                or Path(raw_path).suffix.casefold() == extension
            )
            hidden = not (matches_query and matches_type)
            item.setHidden(hidden)
            if hidden and self.files.currentItem() is item:
                self.files.setCurrentRow(-1)
            if not hidden:
                visible_count += 1

        self.status.setText(
            f"Exibindo {visible_count} de {self.total_files} arquivo(s). "
            f"Tipo: {self.type_filter.currentText()}."
        )
        if self.restore_warning:
            self.status.setText(f"{self.status.text()} {self.restore_warning}")
        if visible_count == 0:
            self.status.setText(
                f"Nenhum arquivo encontrado para: {self.type_filter.currentText()}."
            )

    def _current_path(self) -> Path | None:
        item = self.files.currentItem()
        if item is None or item.isHidden():
            return None
        return Path(item.data(Qt.ItemDataRole.UserRole))

    def _preview_selection_changed(self, current, _previous) -> None:
        """Preview audio whenever keyboard navigation changes the selected item."""
        if current is not None and not current.isHidden():
            self._preview_item(current)

    def _preview_item(self, item) -> None:
        """Play an audio item selected by mouse or keyboard, without restarting it twice."""
        if item is None or item.isHidden():
            return

        path = Path(item.data(Qt.ItemDataRole.UserRole))
        if path.suffix.lower() == ".safetensors":
            self.player.stop()
            self.player.setSource(QUrl())
            self.status.setText(
                "Este é um estado de voz, não uma gravação. Selecione um arquivo de áudio."
            )
            return

        source = QUrl.fromLocalFile(str(path.resolve()))
        if (
            self.player.source() == source
            and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        ):
            self.status.setText(f"Reproduzindo: {path.name}")
            return

        self.player.setSource(source)
        self.player.play()
        self.status.setText(f"Reproduzindo: {path.name}")

    def _preview(self) -> None:
        """Keep the explicit preview action compatible with existing callers."""
        item = self.files.currentItem()
        if item is None:
            self.status.setText("Selecione um arquivo primeiro.")
            return
        self._preview_item(item)

    def _accept_selection(self) -> None:
        path = self._current_path()
        if path is None:
            self.status.setText("Selecione um arquivo de voz.")
            return
        self.selected_path = path
        self.player.stop()
        self.player.setSource(QUrl())
        self.accept()

    def reject(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())
        super().reject()


class FavoriteVoicesDialog(QDialog):
    """Quick picker for saved voice favorites with one-click audio preview."""

    def __init__(
        self,
        favorites: dict[str, str],
        player: QMediaPlayer,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.player = player
        self.selected_path: Path | None = None
        self.setWindowTitle("Vozes favoritas")
        self.resize(680, 440)

        layout = QVBoxLayout(self)
        description = QLabel(
            "Clique em uma voz para ouvir. Dê duplo clique ou use "
            "“Usar esta voz” para torná-la a voz padrão."
        )
        description.setWordWrap(True)
        description.setObjectName("muted")
        layout.addWidget(description)

        self.files = QListWidget()
        self.files.itemDoubleClicked.connect(lambda _item: self._accept_selection())
        layout.addWidget(self.files, 1)

        self.status = QLabel()
        self.status.setObjectName("muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        actions = QHBoxLayout()
        stop_button = QPushButton("Parar reprodução")
        stop_button.clicked.connect(self.player.stop)
        actions.addWidget(stop_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Usar esta voz")
        buttons.accepted.connect(self._accept_selection)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        for raw_path, alias in sorted(
            favorites.items(), key=lambda entry: entry[1].casefold()
        ):
            path = Path(raw_path)
            label = f"★ {alias}  —  {path.name}"
            if not path.is_file():
                label += "  (arquivo não encontrado)"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, str(path))
            item.setToolTip(f"{alias}\n{path}")
            self.files.addItem(item)

        if favorites:
            self.files.setCurrentRow(0)
            self.status.setText(f"{len(favorites)} voz(es) favorita(s).")
        else:
            self.status.setText(
                "Você ainda não adicionou vozes aos favoritos. "
                "Use Configurações → Idioma e voz para adicionar a voz selecionada."
            )

        # Connect after restoring the first row so opening the dialog stays silent.
        self.files.itemClicked.connect(self._preview_item)
        self.files.currentItemChanged.connect(self._preview_selection_changed)

    def _preview_selection_changed(self, current, _previous) -> None:
        if current is not None:
            self._preview_item(current)

    def _preview_item(self, item) -> None:
        if item is None:
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        if not path.is_file():
            self.player.stop()
            self.player.setSource(QUrl())
            self.status.setText(f"Arquivo não encontrado: {path}")
            return
        if path.suffix.lower() == ".safetensors":
            self.player.stop()
            self.player.setSource(QUrl())
            self.status.setText(
                "Estado .safetensors selecionado. Ele pode ser usado como voz, "
                "mas não contém áudio para prévia."
            )
            return

        source = QUrl.fromLocalFile(str(path.resolve()))
        if (
            self.player.source() == source
            and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        ):
            self.status.setText(f"Reproduzindo: {path.name}")
            return
        self.player.setSource(source)
        self.player.play()
        self.status.setText(f"Reproduzindo: {path.name}")

    def _accept_selection(self) -> None:
        item = self.files.currentItem()
        if item is None:
            self.status.setText("Selecione uma voz favorita.")
            return
        path = Path(item.data(Qt.ItemDataRole.UserRole))
        if not path.is_file():
            self.status.setText("O arquivo desta voz não foi encontrado.")
            return
        self.selected_path = path
        self.player.stop()
        self.player.setSource(QUrl())
        self.accept()

    def reject(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())
        super().reject()


class HotkeyBridge(QObject):
    activated = Signal()
    command_received = Signal(str)


class ClipboardAudioDialog(QDialog):
    """Compact playback window for chunked clipboard audio."""

    closed = Signal()
    saved = Signal(str, object)

    def __init__(
        self,
        player: QMediaPlayer,
        output_directory: str,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.player = player
        self.output_directory = output_directory
        self.temp_dir: Path | None = None
        self.audio_path: Path | None = None
        self.performance: AudioPerformance | None = None
        self.chunk_paths: list[Path] = []
        self.next_chunk_index = 0
        self.current_chunk_path: Path | None = None
        self.generation_finished = False
        self.play_requested = False
        self.waiting_for_chunk = False
        self.playing_final_audio = False
        self.is_closing = False
        self.generation_started_at: float | None = None
        self.performance_timer = QTimer(self)
        self.performance_timer.setInterval(500)
        self.performance_timer.timeout.connect(self._update_performance_elapsed)

        self.setWindowTitle("Joe — Texto copiado para áudio")
        # Tool windows are commonly omitted from the task switcher/taskbar on Linux.
        # Use a regular window so the shortcut window has its own Joe icon.
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowIcon(QApplication.windowIcon())
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(480, 330)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        self.preview_label = QLabel("Texto copiado")
        self.preview_label.setWordWrap(True)
        self.preview_label.setObjectName("muted")
        layout.addWidget(self.preview_label)

        self.status_label = QLabel("Preparando trechos de áudio…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.performance_label = QLabel(
            "Medição de velocidade feita pelo Joe aparecerá aqui."
        )
        self.performance_label.setObjectName("muted")
        self.performance_label.setWordWrap(True)
        layout.addWidget(self.performance_label)

        buttons = QHBoxLayout()
        self.play_button = QPushButton("Reproduzir")
        self.play_button.setObjectName("primaryButton")
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._play)
        self.pause_button = QPushButton("Pausar")
        self.pause_button.setEnabled(False)
        self.pause_button.clicked.connect(self._pause)
        self.save_button = QPushButton("Salvar áudio…")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)
        self.close_button = QPushButton("Fechar")
        self.close_button.clicked.connect(self.close)

        buttons.addWidget(self.play_button)
        buttons.addWidget(self.pause_button)
        buttons.addWidget(self.save_button)
        buttons.addWidget(self.close_button)
        layout.addLayout(buttons)

        self.player.mediaStatusChanged.connect(self._on_media_status_changed)

    def set_text_preview(self, text: str) -> None:
        compact = " ".join(text.split())
        if len(compact) > 260:
            compact = compact[:257].rstrip() + "…"
        self.preview_label.setText(compact)

    def start_measurement(self, remote: bool = False) -> None:
        """Inicia o cronômetro do desktop para a geração em blocos."""
        self.generation_started_at = time.perf_counter()
        self.performance_label.setText(
            "Medição do Joe neste computador — "
            f"{format_elapsed_status(0)}"
        )
        self.performance_timer.start()

    def _update_performance_elapsed(self) -> None:
        if self.generation_started_at is None:
            self.performance_timer.stop()
            return
        elapsed = time.perf_counter() - self.generation_started_at
        self.performance_label.setText(
            "Medição do Joe neste computador — "
            f"{format_elapsed_status(elapsed)}"
        )

    def set_progress(self, message: str) -> None:
        if not self.is_closing:
            self.status_label.setText(message)

    def add_chunk(self, index: int, audio_path: Path, total: int) -> None:
        if self.is_closing:
            return
        self.chunk_paths.append(audio_path)
        self.play_button.setEnabled(True)
        if not self.generation_finished and not self.play_requested:
            self.status_label.setText(
                f"Trecho {index} de {total} pronto. Gerando os próximos…"
            )
        if self.waiting_for_chunk and self.play_requested:
            self._play_next_chunk()

    def set_ready(self, result: dict) -> None:
        self.performance_timer.stop()
        self.generation_started_at = None
        self.audio_path = Path(result["path"])
        self.generation_finished = True
        performance = result.get("performance")
        self.performance = performance if isinstance(performance, AudioPerformance) else None
        if isinstance(performance, AudioPerformance):
            self.performance_label.setText(format_performance_report(performance))
        else:
            self.performance_label.setText(
                "Áudio pronto; não foi possível calcular todas as métricas."
            )
        self.play_button.setEnabled(bool(self.chunk_paths))
        self.save_button.setEnabled(True)
        if not self.play_requested:
            self.status_label.setText(
                "Áudio completo pronto. Você pode reproduzir ou salvar o WAV."
            )
        elif self.waiting_for_chunk:
            self._play_next_chunk()

    def set_failed(self, message: str) -> None:
        self.performance_timer.stop()
        self.generation_started_at = None
        self.performance_label.setText(
            "Medição interrompida porque a geração não foi concluída."
        )
        self.generation_finished = True
        self.status_label.setText(
            f"Falha ao gerar o áudio completo: {message}. "
            "Os trechos já gerados ainda podem ser reproduzidos."
        )
        self.play_button.setEnabled(bool(self.chunk_paths))
        self.save_button.setEnabled(False)

    def _play(self) -> None:
        if not self.chunk_paths:
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
            self.play_requested = not self.playing_final_audio
            self.player.play()
            self.pause_button.setEnabled(True)
            return
        if self.generation_finished and self.audio_path and self.audio_path.is_file():
            self.play_requested = False
            self.waiting_for_chunk = False
            self.playing_final_audio = True
            self.current_chunk_path = None
            self.player.setSource(QUrl.fromLocalFile(str(self.audio_path.resolve())))
            self.player.play()
            self.pause_button.setEnabled(True)
            self.status_label.setText("Reproduzindo o WAV completo unido, sem troca entre trechos…")
            return
        self.play_requested = True
        if self.current_chunk_path and self.player.source().isValid() and (
            self.player.mediaStatus() != QMediaPlayer.MediaStatus.EndOfMedia
        ):
            self.player.play()
            self.pause_button.setEnabled(True)
            return
        self._play_next_chunk()

    def _play_next_chunk(self) -> None:
        if not self.play_requested or self.is_closing:
            return
        if self.next_chunk_index < len(self.chunk_paths):
            path = self.chunk_paths[self.next_chunk_index]
            self.next_chunk_index += 1
            self.current_chunk_path = path
            self.waiting_for_chunk = False
            self.player.setSource(QUrl.fromLocalFile(str(path.resolve())))
            self.player.play()
            self.pause_button.setEnabled(True)
            if not self.generation_finished:
                self.status_label.setText(
                    f"Reproduzindo trecho {self.next_chunk_index}; "
                    "os próximos continuam sendo gerados…"
                )
            return

        if self.generation_finished:
            self.play_requested = False
            self.waiting_for_chunk = False
            self.pause_button.setEnabled(False)
            self.status_label.setText("Reprodução concluída.")
            return

        self.waiting_for_chunk = True
        self.pause_button.setEnabled(False)
        self.status_label.setText(
            "Aguardando a geração do próximo trecho para continuar a reprodução…"
        )
        self.player.stop()
        self.player.setSource(QUrl())

    def _on_media_status_changed(self, status) -> None:
        if status != QMediaPlayer.MediaStatus.EndOfMedia:
            return
        if self.playing_final_audio:
            self.playing_final_audio = False
            self.pause_button.setEnabled(False)
            self.status_label.setText("Reprodução concluída.")
        elif self.play_requested:
            self._play_next_chunk()

    def _pause(self) -> None:
        self.play_requested = False
        self.player.pause()
        self.pause_button.setEnabled(False)

    def _save(self) -> None:
        if not self.audio_path or not self.audio_path.is_file():
            return

        initial_directory = (
            Path(self.output_directory).expanduser()
            if self.output_directory.strip()
            else Path.home()
        )
        if not initial_directory.is_dir():
            initial_directory = Path.home()
        suggested = initial_directory / "joe_clipboard_audio.wav"
        output, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar áudio gerado",
            str(suggested),
            "Áudio WAV (*.wav)",
        )
        if not output:
            return
        if not output.lower().endswith(".wav"):
            output += ".wav"

        destination = Path(output).expanduser()
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(self.audio_path, destination)
        except OSError as exc:
            QMessageBox.critical(self, "Erro ao salvar áudio", str(exc))
            return
        self.status_label.setText(f"Cópia salva em:\n{destination}")
        self.saved.emit(str(destination), self.performance)

    def closeEvent(self, event) -> None:
        self.is_closing = True
        self.performance_timer.stop()
        self.generation_started_at = None
        self.play_requested = False
        current_source = self.player.source().toLocalFile()
        chunk_sources = {str(path.resolve()) for path in self.chunk_paths}
        if current_source and (
            current_source in chunk_sources
            or (self.audio_path and current_source == str(self.audio_path.resolve()))
        ):
            self.player.stop()
            self.player.setSource(QUrl())
        self.closed.emit()
        super().closeEvent(event)


class JoeWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Joe — Audio Studio")
        self.setWindowIcon(QApplication.windowIcon())
        self.resize(980, 740)
        self.setMinimumSize(760, 600)

        self.tts = PocketTTSService()
        self.asr = ParakeetReduxService()
        self.app_settings = QSettings("g4b3r-desktop", "JoeAudioStudio")
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self.active_worker: FunctionWorker | None = None
        self.generation_queue = GenerationQueue()
        self.active_generation_worker: TTSGenerationWorker | None = None
        self.generation_queue_dialog: GenerationQueueDialog | None = None
        self.clipboard_generation_worker: FunctionWorker | None = None
        self.clipboard_audio_window: ClipboardAudioDialog | None = None
        self.clipboard_temp_dir: Path | None = None
        self.hotkey_listener = None
        self.local_command_server: LocalCommandServer | None = None
        self.clipboard_only_mode = False
        self.hotkey_bridge = HotkeyBridge(self)
        self.hotkey_bridge.activated.connect(self._generate_audio_from_clipboard)
        self.hotkey_bridge.command_received.connect(self._handle_local_command)
        self.hotkey_monitor_timer = QTimer(self)
        self.hotkey_monitor_timer.setInterval(2500)
        self.hotkey_monitor_timer.timeout.connect(self._check_global_hotkey)

        # Health checks monitor connectivity only; they do not keep Colab awake.
        self.colab_monitor_pool = QThreadPool(self)
        self.colab_monitor_pool.setMaxThreadCount(1)
        self.colab_monitor_worker: FunctionWorker | None = None
        self.colab_monitor_manual = False
        self.colab_monitor_timer = QTimer(self)
        self.colab_monitor_timer.setInterval(60_000)
        self.colab_monitor_timer.timeout.connect(self._refresh_colab_status)

        self.generated_audio: Path | None = None
        self.tts_generation_started_at: float | None = None
        self.transcript_segments: list[dict] = []
        self.transcript_audio_path = ""
        self.transcription_backend = ""
        self.pronunciation_dictionary = self._load_pronunciation_dictionary()
        self.pronunciation_rules = load_pronunciation_rules(
            self.app_settings.value("pronunciation_rules_v1", "[]")
        )
        self.favorite_voices = load_voice_favorites(
            self.app_settings.value("favorite_voices", "{}")
        )
        self.voice_profiles = load_voice_profiles(
            self.app_settings.value("tts_voice_profiles", "[]")
        )
        history_directory = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.AppDataLocation
        )
        if not history_directory:
            history_directory = str(Path.home() / ".local" / "share" / "joe-audio-studio")
        self.audio_history = AudioHistoryStore(
            Path(history_directory) / "audio-history.sqlite3"
        )

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._update_transcript_sync_position)

        self._build_ui()
        self.tts_favorites_button.setText(
            f"★ Favoritos ({len(self.favorite_voices)})"
        )
        self.tts_elapsed_timer = QTimer(self)
        self.tts_elapsed_timer.setInterval(500)
        self.tts_elapsed_timer.timeout.connect(self._update_tts_elapsed)
        self.setStyleSheet(self._stylesheet())
        self.statusBar().showMessage("Pronto — os modelos serão carregados sob demanda.")
        self._start_global_hotkey()
        self._update_colab_monitoring()

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_tts_tab(), "Texto → Áudio")
        tabs.addTab(self._build_asr_tab(), "Áudio → Texto")
        tabs.addTab(self._build_colab_tab(), "Google Colab / API")
        # Add after Colab so the persisted tab index 2 keeps its previous meaning.
        tabs.addTab(self._build_history_tab(), "Histórico")
        # The new player is appended so saved indices of existing tabs stay stable.
        tabs.addTab(self._build_player_tab(), "Player avançado")
        tabs.setCurrentIndex(self.app_settings.value("last_tab", 0, type=int))
        tabs.currentChanged.connect(self._on_main_tab_changed)
        self.setCentralWidget(tabs)

        bar = QStatusBar()
        self.setStatusBar(bar)
        self.status_label = QLabel("Pronto")
        bar.addPermanentWidget(self.status_label)

        file_menu = self.menuBar().addMenu("Arquivo")
        exit_action = QAction("Sair", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        clipboard_action = QAction(
            "Gerar áudio do texto copiado (Ctrl+Shift+Espaço)", self
        )
        clipboard_action.triggered.connect(self._generate_audio_from_clipboard)
        file_menu.addAction(clipboard_action)
        restart_hotkey_action = QAction("Reativar atalho global", self)
        restart_hotkey_action.triggered.connect(self._restart_global_hotkey)
        file_menu.addAction(restart_hotkey_action)

        help_menu = self.menuBar().addMenu("Ajuda")
        about_action = QAction("Sobre o Joe", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_colab_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Onde executar os modelos?")
        title.setObjectName("pageTitle")
        description = QLabel(
            "Escolha processamento local ou use uma sessão do Google Colab como servidor remoto."
        )
        description.setObjectName("muted")
        description.setWordWrap(True)

        self.processing_mode = QComboBox()
        self.processing_mode.addItem("Neste computador (Local)", "local")
        self.processing_mode.addItem("Google Colab (API remota)", "colab")
        saved_mode = str(self.app_settings.value("processing_mode", "local"))
        mode_index = self.processing_mode.findData(saved_mode)
        self.processing_mode.setCurrentIndex(max(0, mode_index))
        self.processing_mode.currentIndexChanged.connect(self._save_colab_settings)

        self.colab_url = QLineEdit(str(self.app_settings.value("colab_url", "")))
        self.colab_url.setPlaceholderText("https://seu-tunel.trycloudflare.com")
        self.colab_url.textChanged.connect(
            lambda value: self._save_text_setting("colab_url", value)
        )

        self.colab_token = QLineEdit(
            str(self.app_settings.value("colab_token", ""))
        )
        self.colab_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.colab_token.setPlaceholderText("Cole o token gerado pelo notebook")
        self.colab_token.textChanged.connect(
            lambda value: self._save_text_setting("colab_token", value)
        )
        self.colab_token.editingFinished.connect(self._update_colab_monitoring)

        form = QFormLayout()
        form.addRow("Modo de execução", self.processing_mode)
        form.addRow("URL da API", self.colab_url)
        form.addRow("Token de acesso", self.colab_token)

        self.colab_test_button = QPushButton("Testar conexão com o Colab")
        self.colab_test_button.setObjectName("primaryButton")
        self.colab_test_button.clicked.connect(self._test_colab_connection)

        self.colab_open_button = QPushButton("Abrir notebook no Google Colab")
        self.colab_open_button.clicked.connect(self._open_colab_notebook)

        self.colab_status = QLabel("Estado da API: não configurado")
        self.colab_status.setObjectName("muted")
        self.colab_status.setWordWrap(True)

        self.colab_info = QLabel(
            "1. Abra o notebook no Google Colab e execute as células.\n"
            "2. Copie a URL trycloudflare.com e o token exibidos no final.\n"
            "3. Cole os dois campos acima e selecione Google Colab no modo.\n\n"
            "O Joe verifica a conexão a cada 60 segundos enquanto o modo Colab está selecionado. "
            "Essa verificação apenas detecta desconexões: não impede o Colab de encerrar o runtime "
            "por inatividade ou por seus limites de sessão.\n"
            "A URL e o token são lembrados localmente entre execuções. O token fica salvo "
            "nas configurações do usuário sem criptografia; não compartilhe o arquivo de "
            "configuração do Joe."
        )
        self.colab_info.setObjectName("muted")
        self.colab_info.setWordWrap(True)

        self.colab_url.editingFinished.connect(self._save_colab_settings)

        buttons = QHBoxLayout()
        buttons.addWidget(self.colab_test_button)
        buttons.addWidget(self.colab_open_button)
        buttons.addStretch(1)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(form)
        layout.addLayout(buttons)
        layout.addWidget(self.colab_status)
        layout.addWidget(self.colab_info)
        layout.addStretch(1)

        self._update_mode_widgets()
        return page

    def _save_colab_settings(self, *_args) -> None:
        if hasattr(self, "processing_mode"):
            self.app_settings.setValue("processing_mode", self.processing_mode.currentData())
        if hasattr(self, "colab_url"):
            self.app_settings.setValue("colab_url", self.colab_url.text().strip())
        self._update_mode_widgets()
        self._update_colab_monitoring()

    def _update_mode_widgets(self) -> None:
        if hasattr(self, "asr_device") and hasattr(self, "processing_mode"):
            self.asr_device.setEnabled(self.processing_mode.currentData() == "local")

    def _colab_client(self) -> ColabAPIClient:
        return ColabAPIClient(self.colab_url.text(), self.colab_token.text())

    def _update_colab_monitoring(self, *_args) -> None:
        if not hasattr(self, "processing_mode") or not hasattr(self, "colab_status"):
            return

        if self.processing_mode.currentData() != "colab":
            self.colab_monitor_timer.stop()
            self.colab_status.setText("Estado da API: modo local ativo; conexão remota não necessária.")
            return

        if not self.colab_url.text().strip() or not self.colab_token.text().strip():
            self.colab_monitor_timer.stop()
            self.colab_status.setText(
                "Estado da API: aguardando a URL HTTPS e o token do notebook."
            )
            return

        if not self.colab_monitor_timer.isActive():
            self.colab_monitor_timer.start()
        self._refresh_colab_status()

    def _refresh_colab_status(self) -> None:
        if self.processing_mode.currentData() != "colab":
            return
        if not self.colab_url.text().strip() or not self.colab_token.text().strip():
            return
        self._start_colab_check(manual=False)

    def _start_colab_check(self, manual: bool) -> None:
        if self.colab_monitor_worker is not None:
            if manual:
                self.statusBar().showMessage("A verificação da API já está em andamento.", 4000)
            return
        try:
            client = self._colab_client()
        except ValueError as exc:
            self.colab_status.setText(f"Estado da API: configuração incompleta — {exc}")
            if manual:
                QMessageBox.information(self, "Configuração incompleta", str(exc))
            return

        self.colab_monitor_manual = manual
        self.colab_status.setText("Estado da API: verificando conexão…")
        worker = FunctionWorker(client.test_connection)
        self.colab_monitor_worker = worker
        worker.signals.finished.connect(self._colab_connection_finished)
        worker.signals.failed.connect(self._colab_monitor_failed)
        self.colab_monitor_pool.start(worker)

    def _test_colab_connection(self) -> None:
        self._start_colab_check(manual=True)

    def _colab_connection_finished(self, result: object) -> None:
        self.colab_monitor_worker = None
        self.colab_monitor_manual = False
        data = result if isinstance(result, dict) else {}
        gpu = data.get("gpu_name") or ("CUDA disponível" if data.get("cuda_available") else "CPU")
        checked_at = QTime.currentTime().toString("HH:mm:ss")
        self.colab_status.setText(f"API online — última verificação às {checked_at}.")
        self.colab_info.setText(
            f"Conexão confirmada. Servidor: {data.get('service', 'Joe Colab API')}\n"
            f"ASR: {data.get('asr', 'Parakeet Redux')}\n"
            f"GPU: {gpu}\n"
            f"Parakeet carregado: {'sim' if data.get('asr_loaded') else 'ainda não'}\n\n"
            "O monitoramento confirma conectividade, mas não mantém o runtime do Colab ativo."
        )
        self.statusBar().showMessage("API do Google Colab conectada.", 5000)

    def _colab_monitor_failed(self, message: str) -> None:
        manual = self.colab_monitor_manual
        self.colab_monitor_worker = None
        self.colab_monitor_manual = False
        detail = message.split("\n\n", 1)[0].strip() or "Não foi possível alcançar a API."
        checked_at = QTime.currentTime().toString("HH:mm:ss")
        self.colab_status.setText(
            f"API indisponível — {detail} (verificado às {checked_at}). "
            "Confira a sessão do Colab e a URL/token do túnel."
        )
        self.statusBar().showMessage("API do Colab indisponível.", 6000)
        if manual:
            QMessageBox.warning(
                self,
                "API do Colab indisponível",
                f"{detail}\n\n"
                "Se o runtime foi encerrado, execute novamente o notebook e informe "
                "a nova URL do túnel e o token.",
            )

    def _open_colab_notebook(self) -> None:
        url = (
            "https://colab.research.google.com/github/g4b3r-desktop/joe/"
            "blob/feature/joe-studio-improvements/colab/Joe_Colab_API.ipynb"
        )
        QDesktopServices.openUrl(QUrl(url))

    def _build_tts_tab(self) -> QWidget:
        """Build the frequent text-to-speech workflow."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Transforme texto em fala")
        title.setObjectName("pageTitle")

        description = QLabel(
            "Digite ou cole o texto e gere o áudio. "
            "Ctrl+Shift+Espaço converte diretamente o texto copiado."
        )
        description.setObjectName("muted")
        description.setWordWrap(True)

        self.tts_text = QTextEdit()
        self.tts_text.setPlaceholderText(
            "Digite ou cole aqui o texto que deseja transformar em áudio..."
        )
        self.tts_text.setPlainText(str(self.app_settings.value("tts_text", "")))
        self.tts_text.setMinimumHeight(280)
        self.tts_text.textChanged.connect(
            lambda: self._save_text_setting("tts_text", self.tts_text.toPlainText())
        )

        self.tts_generate_button = QPushButton("Gerar áudio WAV")
        self.tts_generate_button.setObjectName("primaryButton")
        self.tts_generate_button.clicked.connect(self._generate_audio)

        self.tts_queue_button = QPushButton("Fila de geração (0)")
        self.tts_queue_button.setToolTip(
            "Adicionar textos, acompanhar tarefas, cancelar, repetir ou priorizar"
        )
        self.tts_queue_button.clicked.connect(self._open_generation_queue)

        self.tts_play_button = QPushButton("Reproduzir último áudio")
        self.tts_play_button.setEnabled(False)
        self.tts_play_button.clicked.connect(self._play_generated_audio)

        self.tts_favorites_button = QPushButton("★ Favoritos")
        self.tts_favorites_button.setToolTip(
            "Escolher rapidamente uma voz salva nos favoritos"
        )
        self.tts_favorites_button.clicked.connect(self._open_favorite_voices)

        self.tts_settings_button = QPushButton("⚙ Configurações")
        self.tts_settings_button.setToolTip(
            "Idioma, voz, pronúncia, pasta de saída e opções do atalho"
        )
        self.tts_settings_button.clicked.connect(self._open_tts_settings)

        action_row = QHBoxLayout()
        action_row.addWidget(self.tts_generate_button)
        action_row.addWidget(self.tts_queue_button)
        action_row.addWidget(self.tts_play_button)
        action_row.addWidget(self.tts_favorites_button)
        action_row.addStretch(1)
        action_row.addWidget(self.tts_settings_button)

        self.tts_info = QLabel(
            "O texto original permanece intacto. As opções de voz e pronúncia "
            "podem ser alteradas em Configurações."
        )
        self.tts_info.setObjectName("muted")
        self.tts_info.setWordWrap(True)

        self.tts_speed_info = QLabel(
            "A medição é feita pelo Joe neste computador. "
            "Após gerar, serão exibidos tempo total, duração do WAV e velocidade efetiva."
        )
        self.tts_speed_info.setObjectName("muted")
        self.tts_speed_info.setWordWrap(True)

        self._build_tts_settings_dialog()

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(self.tts_text, 1)
        layout.addLayout(action_row)
        layout.addWidget(self.tts_info)
        layout.addWidget(self.tts_speed_info)
        return page

    def _build_tts_settings_dialog(self) -> None:
        """Create less frequently changed TTS options in a separate dialog."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Configurações — Texto para áudio")
        available_height = self.screen().availableGeometry().height()
        dialog.resize(760, min(840, max(420, available_height - 80)))
        self.tts_settings_dialog = dialog

        dialog_layout = QVBoxLayout(dialog)
        dialog_layout.setContentsMargins(8, 8, 8, 8)
        scroll_area = QScrollArea(dialog)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        settings_content = QWidget()
        settings_layout = QVBoxLayout(settings_content)
        settings_layout.setContentsMargins(14, 12, 14, 12)
        settings_layout.setSpacing(14)
        scroll_area.setWidget(settings_content)
        dialog_layout.addWidget(scroll_area)

        intro = QLabel(
            "Ajuste aqui as opções que normalmente precisam ser configuradas apenas uma vez. "
            "As alterações são salvas automaticamente."
        )
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        settings_layout.addWidget(intro)

        # Text preparation and pronunciation.
        self.tts_normalization_enabled = QCheckBox(
            "Normalizar números, datas, siglas e abreviações para a fala"
        )
        self.tts_normalization_enabled.setChecked(
            self.app_settings.value("tts_normalization_enabled", True, type=bool)
        )
        self.tts_normalization_enabled.toggled.connect(
            self._save_tts_normalization_setting
        )

        self.tts_dictionary_button = QPushButton("Dicionário de pronúncia…")
        self.tts_dictionary_button.clicked.connect(self._manage_pronunciation_dictionary)
        self.tts_preview_button = QPushButton("Prévia do texto falado")
        self.tts_preview_button.clicked.connect(self._preview_tts_text)

        pronunciation_group = QGroupBox("Texto e pronúncia")
        pronunciation_layout = QVBoxLayout(pronunciation_group)
        pronunciation_layout.addWidget(self.tts_normalization_enabled)
        pronunciation_actions = QHBoxLayout()
        pronunciation_actions.addWidget(self.tts_dictionary_button)
        self.tts_advanced_rules_button = QPushButton("Regras avançadas…")
        self.tts_advanced_rules_button.clicked.connect(
            self._manage_advanced_pronunciation_rules
        )
        pronunciation_actions.addWidget(self.tts_advanced_rules_button)
        pronunciation_actions.addWidget(self.tts_preview_button)
        pronunciation_actions.addStretch(1)
        pronunciation_layout.addLayout(pronunciation_actions)
        settings_layout.addWidget(pronunciation_group)

        # Language and voice selection.
        self.tts_language = QComboBox()
        self.tts_language.addItem("Português", "portuguese")
        self.tts_language.addItem("Inglês", "english")
        self.tts_language.addItem("Espanhol", "spanish")
        self.tts_language.addItem("Francês", "french")
        self.tts_language.addItem("Italiano", "italian")
        self.tts_language.addItem("Alemão", "german")
        self.tts_language.addItem("Neerlandês", "dutch")
        saved_language = str(self.app_settings.value("tts_language", "portuguese"))
        language_index = self.tts_language.findData(saved_language)
        self.tts_language.setCurrentIndex(max(0, language_index))
        self.tts_language.currentIndexChanged.connect(
            lambda: self.app_settings.setValue(
                "tts_language", self.tts_language.currentData()
            )
        )

        self.voice_path = QLineEdit(str(self.app_settings.value("voice_path", "")))
        self.voice_path.setReadOnly(True)
        self.voice_path.setPlaceholderText("Voz padrão do Pocket TTS")
        self.voice_path.textChanged.connect(
            lambda value: self._save_text_setting("voice_path", value)
        )
        self.voice_path.textChanged.connect(self._sync_favorite_editor)

        self.voice_library_directory = QLineEdit(
            str(self.app_settings.value("voice_library_directory", ""))
        )
        self.voice_library_directory.setPlaceholderText(
            "Pasta principal; a busca inclui todas as subpastas"
        )
        self.voice_library_directory.textChanged.connect(
            lambda value: self._save_text_setting("voice_library_directory", value)
        )

        self.voice_library_button = QPushButton("Escolher pasta…")
        self.voice_library_button.clicked.connect(self._choose_voice_library_directory)

        library_row = QWidget()
        library_layout = QHBoxLayout(library_row)
        library_layout.setContentsMargins(0, 0, 0, 0)
        library_layout.addWidget(self.voice_library_directory, 1)
        library_layout.addWidget(self.voice_library_button)

        self.voice_button = QPushButton("Biblioteca…")
        self.voice_button.clicked.connect(self._choose_voice)
        self.voice_preview_button = QPushButton("▶ Ouvir voz")
        self.voice_preview_button.clicked.connect(self._preview_selected_voice)
        self.voice_prepare_button = QPushButton("Preparar voz (.safetensors)…")
        self.voice_prepare_button.setToolTip(
            "Converte o áudio de referência em um estado de voz reutilizável do Pocket TTS."
        )
        self.voice_prepare_button.clicked.connect(self._prepare_voice_state)

        voice_row = QWidget()
        voice_layout = QHBoxLayout(voice_row)
        voice_layout.setContentsMargins(0, 0, 0, 0)
        voice_layout.addWidget(self.voice_path, 1)
        voice_layout.addWidget(self.voice_button)
        voice_layout.addWidget(self.voice_preview_button)
        voice_layout.addWidget(self.voice_prepare_button)

        voice_group = QGroupBox("Idioma e voz")
        voice_form = QFormLayout(voice_group)
        voice_form.addRow("Idioma da fala", self.tts_language)
        voice_form.addRow("Pasta da biblioteca", library_row)
        voice_form.addRow("Voz padrão / referência", voice_row)
        settings_layout.addWidget(voice_group)

        # Named voice profiles are independent from favorites and pronunciation rules.
        profile_group = QGroupBox("Perfis de voz")
        profile_layout = QFormLayout(profile_group)
        self.voice_profile_combo = QComboBox()
        self.voice_profile_combo.currentIndexChanged.connect(
            self._on_voice_profile_selected
        )
        self.voice_profile_name = QLineEdit()
        self.voice_profile_name.setPlaceholderText("Ex.: Narradora em português")
        profile_actions = QHBoxLayout()
        self.voice_profile_apply_button = QPushButton("Aplicar perfil")
        self.voice_profile_apply_button.clicked.connect(self._apply_voice_profile)
        self.voice_profile_save_button = QPushButton("Salvar perfil")
        self.voice_profile_save_button.clicked.connect(self._save_voice_profile)
        self.voice_profile_delete_button = QPushButton("Excluir perfil")
        self.voice_profile_delete_button.clicked.connect(self._delete_voice_profile)
        profile_actions.addWidget(self.voice_profile_apply_button)
        profile_actions.addWidget(self.voice_profile_save_button)
        profile_actions.addWidget(self.voice_profile_delete_button)
        profile_actions.addStretch(1)
        profile_layout.addRow("Perfis salvos", self.voice_profile_combo)
        profile_layout.addRow("Nome do perfil", self.voice_profile_name)
        profile_layout.addRow("", profile_actions)
        settings_layout.addWidget(profile_group)
        self._refresh_voice_profile_combo(
            str(self.app_settings.value("active_tts_voice_profile", ""))
        )

        favorite_group = QGroupBox("Favoritos")
        favorite_layout = QFormLayout(favorite_group)
        self.favorite_voice_alias = QLineEdit()
        self.favorite_voice_alias.setPlaceholderText(
            "Ex.: Narradora, Voz grave, Personagem principal…"
        )
        self.favorite_voice_alias.setToolTip(
            "Apelido usado para localizar esta voz no botão Favoritos da tela principal."
        )
        favorite_layout.addRow("Apelido da voz selecionada", self.favorite_voice_alias)

        favorite_actions = QHBoxLayout()
        self.favorite_voice_save_button = QPushButton("Adicionar aos favoritos")
        self.favorite_voice_save_button.clicked.connect(
            self._save_current_voice_favorite
        )
        self.favorite_voice_remove_button = QPushButton("Remover dos favoritos")
        self.favorite_voice_remove_button.clicked.connect(
            self._remove_current_voice_favorite
        )
        favorite_actions.addWidget(self.favorite_voice_save_button)
        favorite_actions.addWidget(self.favorite_voice_remove_button)
        favorite_actions.addStretch(1)
        favorite_layout.addRow("", favorite_actions)
        settings_layout.addWidget(favorite_group)
        self._sync_favorite_editor()

        # Output folder and clipboard shortcut settings.
        default_output_dir = QStandardPaths.writableLocation(
            QStandardPaths.StandardLocation.MusicLocation
        )
        if not default_output_dir:
            default_output_dir = str(Path.home() / "Music")
        default_output_dir = str(Path(default_output_dir) / "Joe")
        saved_output_dir = str(
            self.app_settings.value("tts_output_directory", default_output_dir)
        )

        self.tts_output_directory = QLineEdit(saved_output_dir)
        self.tts_output_directory.setPlaceholderText("Pasta onde os WAV serão salvos")
        self.tts_output_directory.textChanged.connect(self._save_tts_output_settings)
        self.tts_output_directory.editingFinished.connect(self._save_tts_output_settings)

        self.tts_browse_output_button = QPushButton("Escolher pasta…")
        self.tts_browse_output_button.clicked.connect(self._choose_tts_output_directory)

        output_folder_row = QWidget()
        output_folder_layout = QHBoxLayout(output_folder_row)
        output_folder_layout.setContentsMargins(0, 0, 0, 0)
        output_folder_layout.addWidget(self.tts_output_directory, 1)
        output_folder_layout.addWidget(self.tts_browse_output_button)

        self.tts_auto_filename = QCheckBox(
            "Salvar automaticamente com nome genérico (joe_audio_data_hora.wav)"
        )
        self.tts_auto_filename.setChecked(
            self.app_settings.value("tts_auto_filename", True, type=bool)
        )
        self.tts_auto_filename.toggled.connect(self._save_tts_output_settings)

        self.clipboard_chunk_size = QComboBox()
        for size in (180, 280, 420, 560):
            self.clipboard_chunk_size.addItem(f"{size} caracteres", size)
        saved_chunk_size = self.app_settings.value(
            "clipboard_chunk_size", 280, type=int
        )
        chunk_size_index = self.clipboard_chunk_size.findData(saved_chunk_size)
        self.clipboard_chunk_size.setCurrentIndex(max(0, chunk_size_index))
        self.clipboard_chunk_size.currentIndexChanged.connect(
            lambda: self.app_settings.setValue(
                "clipboard_chunk_size", self.clipboard_chunk_size.currentData()
            )
        )

        self.history_save_text_check = QCheckBox(
            "Salvar também o texto original associado ao áudio no histórico"
        )
        self.history_save_text_check.setToolTip(
            "Desativado por padrão. Quando ativado, o texto será guardado no banco local "
            "do histórico para pesquisa e recuperação."
        )
        self.history_save_text_check.setChecked(
            self.app_settings.value("history_save_text", False, type=bool)
        )
        self.history_save_text_check.toggled.connect(
            lambda checked: self.app_settings.setValue("history_save_text", bool(checked))
        )

        output_group = QGroupBox("Arquivos, histórico e atalho de texto copiado")
        output_form = QFormLayout(output_group)
        output_form.addRow("Pasta para salvar", output_folder_row)
        output_form.addRow("", self.tts_auto_filename)
        output_form.addRow("Privacidade do histórico", self.history_save_text_check)
        output_form.addRow("Tamanho dos blocos do atalho", self.clipboard_chunk_size)
        settings_layout.addWidget(output_group)

        footer = QHBoxLayout()
        footer.addStretch(1)
        close_button = QPushButton("Fechar")
        close_button.clicked.connect(dialog.accept)
        footer.addWidget(close_button)
        settings_layout.addLayout(footer)

    def _refresh_voice_profile_combo(self, selected_id: str = "") -> None:
        if not hasattr(self, "voice_profile_combo"):
            return
        combo = self.voice_profile_combo
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Selecione um perfil…", "")
        for profile_id, profile in sorted(
            self.voice_profiles.items(),
            key=lambda item: item[1]["name"].casefold(),
        ):
            combo.addItem(profile["name"], profile_id)
        index = combo.findData(selected_id)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.blockSignals(False)
        self._on_voice_profile_selected()

    def _on_voice_profile_selected(self, *_args) -> None:
        if not hasattr(self, "voice_profile_combo"):
            return
        profile_id = str(self.voice_profile_combo.currentData() or "")
        profile = self.voice_profiles.get(profile_id)
        self.voice_profile_name.setText(profile["name"] if profile else "")
        self.voice_profile_delete_button.setEnabled(profile is not None)
        self.voice_profile_apply_button.setEnabled(profile is not None)
        self.voice_profile_save_button.setText(
            "Atualizar perfil" if profile else "Salvar perfil"
        )

    def _save_voice_profiles(self) -> None:
        self.app_settings.setValue(
            "tts_voice_profiles", serialize_voice_profiles(self.voice_profiles)
        )
        self.app_settings.sync()

    def _save_voice_profile(self) -> None:
        name = self.voice_profile_name.text().strip()
        if not name:
            QMessageBox.information(
                self, "Nome do perfil", "Informe um nome para salvar o perfil de voz."
            )
            self.voice_profile_name.setFocus()
            return

        selected_id = str(self.voice_profile_combo.currentData() or "")
        existing_id = next(
            (
                profile_id
                for profile_id, profile in self.voice_profiles.items()
                if profile["name"].casefold() == name.casefold()
            ),
            "",
        )
        profile_id = selected_id if selected_id in self.voice_profiles else (
            existing_id or uuid.uuid4().hex
        )
        self.voice_profiles[profile_id] = {
            "id": profile_id,
            "name": name,
            "voice_path": self.voice_path.text().strip(),
            "language": str(self.tts_language.currentData() or "portuguese"),
            "normalization_enabled": self.tts_normalization_enabled.isChecked(),
        }
        self._save_voice_profiles()
        self._refresh_voice_profile_combo(profile_id)
        self.app_settings.setValue("active_tts_voice_profile", profile_id)
        self.statusBar().showMessage(f"Perfil de voz “{name}” salvo.", 5000)

    def _apply_voice_profile(self) -> None:
        profile_id = str(self.voice_profile_combo.currentData() or "")
        profile = self.voice_profiles.get(profile_id)
        if profile is None:
            return

        self.voice_path.setText(profile["voice_path"])
        language_index = self.tts_language.findData(profile["language"])
        if language_index >= 0:
            self.tts_language.setCurrentIndex(language_index)
        self.tts_normalization_enabled.setChecked(
            profile["normalization_enabled"]
        )
        self.app_settings.setValue("active_tts_voice_profile", profile_id)
        self.app_settings.sync()
        self._sync_favorite_editor()
        self.statusBar().showMessage(
            f"Perfil “{profile['name']}” aplicado: voz, idioma e normalização atualizados.",
            6000,
        )

    def _delete_voice_profile(self) -> None:
        profile_id = str(self.voice_profile_combo.currentData() or "")
        profile = self.voice_profiles.get(profile_id)
        if profile is None:
            return
        answer = QMessageBox.question(
            self,
            "Excluir perfil de voz",
            f"Excluir o perfil “{profile['name']}”? O arquivo de voz não será apagado.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.voice_profiles.pop(profile_id, None)
        self._save_voice_profiles()
        if str(self.app_settings.value("active_tts_voice_profile", "")) == profile_id:
            self.app_settings.remove("active_tts_voice_profile")
        self._refresh_voice_profile_combo()
        self.statusBar().showMessage(
            f"Perfil “{profile['name']}” excluído. O arquivo de voz foi preservado.",
            5000,
        )

    def _load_favorite_voices(self) -> dict[str, str]:
        """Reload the saved voice favorites from QSettings."""
        return load_voice_favorites(self.app_settings.value("favorite_voices", "{}"))

    def _save_favorite_voices(self) -> None:
        self.app_settings.setValue(
            "favorite_voices",
            serialize_voice_favorites(self.favorite_voices),
        )
        self.app_settings.sync()
        self.tts_favorites_button.setText(
            f"★ Favoritos ({len(self.favorite_voices)})"
        )

    def _sync_favorite_editor(self, *_args) -> None:
        if not hasattr(self, "favorite_voice_alias"):
            return
        current_path = self.voice_path.text().strip()
        key = canonical_voice_path(current_path) if current_path else ""
        alias = self.favorite_voices.get(key, "")
        self.favorite_voice_alias.blockSignals(True)
        self.favorite_voice_alias.setText(alias)
        self.favorite_voice_alias.blockSignals(False)
        is_favorite = bool(key and key in self.favorite_voices)
        self.favorite_voice_save_button.setText(
            "Atualizar favorito" if is_favorite else "Adicionar aos favoritos"
        )
        self.favorite_voice_remove_button.setEnabled(is_favorite)

    def _save_current_voice_favorite(self) -> None:
        raw_path = self.voice_path.text().strip()
        if not raw_path:
            QMessageBox.information(
                self,
                "Nenhuma voz selecionada",
                "Escolha uma voz na Biblioteca antes de adicioná-la aos favoritos.",
            )
            return

        path = Path(raw_path).expanduser()
        if not path.is_file():
            QMessageBox.warning(
                self,
                "Arquivo de voz não encontrado",
                f"Não foi possível localizar o arquivo:\n{path}",
            )
            return

        key = canonical_voice_path(path)
        alias = self.favorite_voice_alias.text().strip() or path.stem
        self.favorite_voices[key] = alias
        self._save_favorite_voices()
        self._sync_favorite_editor()
        self.statusBar().showMessage(
            f"Voz “{alias}” adicionada aos favoritos.", 5000
        )

    def _remove_current_voice_favorite(self) -> None:
        raw_path = self.voice_path.text().strip()
        key = canonical_voice_path(raw_path) if raw_path else ""
        alias = self.favorite_voices.pop(key, None)
        if alias is None:
            return
        self._save_favorite_voices()
        self._sync_favorite_editor()
        self.statusBar().showMessage(f"Voz “{alias}” removida dos favoritos.", 4000)

    def _open_favorite_voices(self) -> None:
        self.favorite_voices = self._load_favorite_voices()
        self.tts_favorites_button.setText(
            f"★ Favoritos ({len(self.favorite_voices)})"
        )
        dialog = FavoriteVoicesDialog(self.favorite_voices, self.player, self)
        if (
            dialog.exec() == QDialog.DialogCode.Accepted
            and dialog.selected_path is not None
        ):
            self.voice_path.setText(str(dialog.selected_path))
            self._sync_favorite_editor()
            self.statusBar().showMessage(
                f"Voz favorita selecionada: {self.favorite_voices.get(canonical_voice_path(dialog.selected_path), dialog.selected_path.stem)}",
                5000,
            )

    def _open_tts_settings(self) -> None:
        """Show one reusable, non-modal TTS settings popup."""
        if not self.tts_settings_dialog.isVisible():
            self.tts_settings_dialog.show()
        self.tts_settings_dialog.raise_()
        self.tts_settings_dialog.activateWindow()

    def _build_history_tab(self) -> QWidget:
        self.history_panel = AudioHistoryPanel(self.audio_history, self.player, self)
        self.history_panel.status_message.connect(
            lambda message: self.statusBar().showMessage(message, 5000)
        )
        return self.history_panel

    def _build_player_tab(self) -> QWidget:
        self.advanced_player = AdvancedAudioPlayer(
            self.player, self.audio_output, self
        )
        self.advanced_player.status_message.connect(
            lambda message: self.statusBar().showMessage(message, 8000)
        )
        return self.advanced_player

    def _on_main_tab_changed(self, index: int) -> None:
        self.app_settings.setValue("last_tab", index)
        if index == 3:
            self._refresh_audio_history()

    def _refresh_audio_history(self) -> None:
        if hasattr(self, "history_panel"):
            self.history_panel.refresh()

    def _record_audio_history(
        self,
        audio_path: str | Path,
        performance: AudioPerformance | None,
        *,
        voice_path: str | None,
        voice_alias: str,
        language: str,
        backend: str,
        original_text: str,
    ) -> None:
        try:
            duration = (
                performance.audio_duration_seconds
                if isinstance(performance, AudioPerformance)
                else wav_duration_seconds(audio_path)
            )
            elapsed = (
                performance.elapsed_seconds
                if isinstance(performance, AudioPerformance)
                else 0.0
            )
            stored_text = (
                original_text
                if hasattr(self, "history_save_text_check")
                and self.history_save_text_check.isChecked()
                else None
            )
            self.audio_history.add_generated_audio(
                audio_path,
                duration_seconds=duration,
                generation_seconds=elapsed,
                voice_path=voice_path,
                voice_alias=voice_alias,
                language=language,
                backend=backend,
                text_content=stored_text,
            )
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(
                f"Áudio gerado, mas não foi possível registrá-lo no histórico: {exc}",
                9000,
            )
            return
        self._refresh_audio_history()

    def _record_task_history(self, task: GenerationTask, result: object) -> None:
        metrics = result if isinstance(result, dict) else {}
        performance = metrics.get("performance")
        self._record_audio_history(
            metrics.get("path", task.output_path),
            performance if isinstance(performance, AudioPerformance) else None,
            voice_path=task.voice_path,
            voice_alias=task.voice_alias,
            language=task.language,
            backend="Google Colab" if task.remote else "Local",
            original_text=task.text,
        )

    def _record_clipboard_history(
        self,
        audio_path: str,
        performance: object,
        copied_text: str,
        language: str,
        voice_path: str | None,
        remote: bool,
    ) -> None:
        voice_key = canonical_voice_path(voice_path) if voice_path else ""
        voice_alias = self.favorite_voices.get(
            voice_key,
            Path(voice_path).stem if voice_path else "Voz padrão",
        )
        self._record_audio_history(
            audio_path,
            performance if isinstance(performance, AudioPerformance) else None,
            voice_path=voice_path,
            voice_alias=voice_alias,
            language=language,
            backend="Google Colab" if remote else "Local",
            original_text=copied_text,
        )

    def _build_asr_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Transcreva áudio em texto")
        title.setObjectName("pageTitle")
        description = QLabel(
            "Selecione uma gravação. O Parakeet Redux detecta o idioma automaticamente."
        )
        description.setObjectName("muted")

        self.audio_path = QLineEdit(
            str(self.app_settings.value("audio_path", ""))
        )
        self.audio_path.setReadOnly(True)
        self.audio_path.setPlaceholderText("Nenhum arquivo selecionado")
        self.audio_path.textChanged.connect(
            lambda value: self._save_text_setting("audio_path", value)
        )
        self.audio_path.textChanged.connect(self._on_transcript_audio_path_changed)
        self.audio_choose_button = QPushButton("Selecionar áudio…")
        self.audio_choose_button.clicked.connect(self._choose_audio)

        file_row = QHBoxLayout()
        file_row.addWidget(self.audio_path, 1)
        file_row.addWidget(self.audio_choose_button)

        self.timestamps_check = QCheckBox("Incluir tempos dos segmentos (quando disponível)")
        self.timestamps_check.setChecked(
            self.app_settings.value("timestamps_enabled", True, type=bool)
        )
        self.timestamps_check.toggled.connect(
            lambda value: self.app_settings.setValue("timestamps_enabled", value)
        )
        self.asr_device = QComboBox()
        self.asr_device.addItem("CPU (preserva a VRAM)", "cpu")
        self.asr_device.addItem("NVIDIA GPU (CUDA)", "cuda")
        saved_device = str(self.app_settings.value("asr_device", "cpu"))
        device_index = self.asr_device.findData(saved_device)
        self.asr_device.setCurrentIndex(max(0, device_index))
        self.asr_device.currentIndexChanged.connect(
            lambda: self.app_settings.setValue("asr_device", self.asr_device.currentData())
        )

        asr_options = QHBoxLayout()
        asr_options.addWidget(self.timestamps_check, 1)
        asr_options.addWidget(QLabel("Dispositivo:"))
        asr_options.addWidget(self.asr_device)

        self.asr_button = QPushButton("Transcrever")
        self.asr_button.setObjectName("primaryButton")
        self.asr_button.clicked.connect(self._transcribe_audio)

        self.transcript = QTextEdit()
        self.transcript.setPlaceholderText(
            "A transcrição aparecerá aqui. Você poderá revisar e editar o texto."
        )
        self.transcript.setPlainText(str(self.app_settings.value("transcript_text", "")))
        self.transcript.textChanged.connect(
            lambda: self._save_text_setting(
                "transcript_text", self.transcript.toPlainText()
            )
        )
        self.transcript.textChanged.connect(self._refresh_transcript_sync_labels)

        self.copy_button = QPushButton("Copiar texto")
        self.copy_button.clicked.connect(self._copy_transcript)
        self.save_txt_button = QPushButton("Salvar TXT…")
        self.save_txt_button.clicked.connect(self._save_transcript)
        self.save_srt_button = QPushButton("Exportar SRT…")
        self.save_srt_button.clicked.connect(self._save_srt)
        self.save_srt_button.setEnabled(False)

        sync_row = QHBoxLayout()
        self.transcript_sync_button = QPushButton("Reproduzir com sincronização")
        self.transcript_sync_button.setEnabled(False)
        self.transcript_sync_button.clicked.connect(self._play_transcript_with_sync)
        self.transcript_sync_info = QLabel(
            "Os trechos com tempo aparecerão aqui após a transcrição."
        )
        self.transcript_sync_info.setObjectName("muted")
        sync_row.addWidget(self.transcript_sync_button)
        sync_row.addWidget(self.transcript_sync_info, 1)

        self.transcript_sync_list = QListWidget()
        self.transcript_sync_list.setMaximumHeight(150)
        self.transcript_sync_list.setToolTip(
            "Clique em um trecho para posicionar o áudio no início daquele segmento."
        )
        self.transcript_sync_list.setEnabled(False)
        self.transcript_sync_list.itemClicked.connect(self._seek_transcript_segment)

        buttons = QHBoxLayout()
        buttons.addWidget(self.asr_button)
        buttons.addStretch(1)
        buttons.addWidget(self.copy_button)
        buttons.addWidget(self.save_txt_button)
        buttons.addWidget(self.save_srt_button)

        self.asr_info = QLabel(
            "O áudio é processado neste computador. A primeira transcrição baixa os pesos "
            "do Parakeet Redux."
        )
        self.asr_info.setObjectName("muted")
        self.asr_info.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addLayout(file_row)
        layout.addLayout(asr_options)
        layout.addLayout(buttons)
        layout.addLayout(sync_row)
        layout.addWidget(self.transcript_sync_list)
        layout.addWidget(self.transcript, 1)
        layout.addWidget(self.asr_info)
        return page

    def _stylesheet(self) -> str:
        return """
        QMainWindow, QWidget {
            background: #17191f;
            color: #e8eaf0;
            font-size: 10pt;
        }
        QMenuBar, QMenu {
            background: #20232b;
            color: #e8eaf0;
        }
        QMenuBar::item:selected, QMenu::item:selected {
            background: #343a49;
        }
        QTabWidget::pane {
            border: 1px solid #343946;
            border-radius: 8px;
            top: -1px;
        }
        QTabBar::tab {
            background: #20232b;
            color: #aeb5c4;
            padding: 11px 22px;
            margin-right: 4px;
            border-top-left-radius: 7px;
            border-top-right-radius: 7px;
        }
        QTabBar::tab:selected {
            background: #303746;
            color: #ffffff;
        }
        QLabel#pageTitle {
            font-size: 19pt;
            font-weight: 650;
            color: #ffffff;
        }
        QLabel#muted {
            color: #a5adbc;
        }
        QGroupBox {
            border: 1px solid #3a4150;
            border-radius: 8px;
            margin-top: 10px;
            padding: 12px 10px 10px;
            font-weight: 600;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 12px;
            padding: 0 5px;
            color: #cbd1dc;
        }
        QTextEdit, QLineEdit, QComboBox {
            background: #20242d;
            color: #f1f3f8;
            border: 1px solid #3a4150;
            border-radius: 7px;
            padding: 9px;
            selection-background-color: #52678e;
        }
        QTextEdit:focus, QLineEdit:focus, QComboBox:focus {
            border: 1px solid #7697d4;
        }
        QPushButton {
            background: #303746;
            color: #f1f3f8;
            border: 1px solid #454e60;
            border-radius: 7px;
            padding: 9px 14px;
        }
        QPushButton:hover {
            background: #3a4354;
        }
        QPushButton:disabled {
            color: #6f7785;
            background: #252a33;
            border-color: #333946;
        }
        QPushButton#primaryButton {
            background: #496ba8;
            border: 1px solid #6287c8;
            color: white;
            font-weight: 600;
            padding: 10px 18px;
        }
        QPushButton#primaryButton:hover {
            background: #587dbb;
        }
        QCheckBox {
            spacing: 8px;
            color: #cbd1dc;
        }
        QStatusBar {
            background: #20232b;
            color: #aeb5c4;
        }
        QStatusBar::item {
            border: none;
        }
        """


    def _save_text_setting(self, key: str, value: str) -> None:
        """Persist editable text and path fields immediately, even if never submitted."""
        self.app_settings.setValue(key, value)

    def closeEvent(self, event) -> None:  # noqa: N802
        # Flush the settings cache so unfinished drafts survive an ordinary app exit.
        self.app_settings.sync()
        super().closeEvent(event)

    def _load_pronunciation_dictionary(self) -> dict[str, str]:
        raw = self.app_settings.value("pronunciation_dictionary", "{}")
        try:
            parsed = json.loads(str(raw))
        except (TypeError, ValueError):
            return {}
        if not isinstance(parsed, dict):
            return {}
        return {
            str(term).strip(): str(pronunciation).strip()
            for term, pronunciation in parsed.items()
            if str(term).strip() and str(pronunciation).strip()
        }

    def _save_pronunciation_dictionary(self, entries: dict[str, str]) -> None:
        self.pronunciation_dictionary = {
            str(term).strip(): str(pronunciation).strip()
            for term, pronunciation in entries.items()
            if str(term).strip() and str(pronunciation).strip()
        }
        self.app_settings.setValue(
            "pronunciation_dictionary",
            json.dumps(self.pronunciation_dictionary, ensure_ascii=False),
        )
        self.statusBar().showMessage(
            f"Dicionário salvo: {len(self.pronunciation_dictionary)} pronúncia(s).", 4000
        )

    def _save_tts_normalization_setting(self, checked: bool) -> None:
        self.app_settings.setValue("tts_normalization_enabled", bool(checked))

    def _prepared_speech_text(self, original_text: str, language: str) -> str:
        return normalize_text_for_tts(
            original_text,
            language=language,
            pronunciation_dictionary=self.pronunciation_dictionary,
            apply_automatic=self.tts_normalization_enabled.isChecked(),
            pronunciation_rules=self.pronunciation_rules,
        )

    def _preview_tts_text(self) -> None:
        original = self.tts_text.toPlainText()
        if not original.strip():
            QMessageBox.information(
                self, "Texto vazio", "Digite ou cole um texto para visualizar a pronúncia."
            )
            return
        try:
            prepared = self._prepared_speech_text(
                original, str(self.tts_language.currentData())
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Erro na normalização", str(exc))
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Prévia do texto enviado ao TTS")
        dialog.resize(640, 420)
        layout = QVBoxLayout(dialog)
        note = QLabel(
            "Esta é a cópia preparada para a fala. O texto original não será alterado."
        )
        note.setWordWrap(True)
        preview = QTextEdit()
        preview.setReadOnly(True)
        preview.setPlainText(prepared)
        layout.addWidget(note)
        layout.addWidget(preview, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_button = buttons.addButton(
            "Copiar texto preparado", QDialogButtonBox.ButtonRole.ActionRole
        )
        copy_button.clicked.connect(lambda: QApplication.clipboard().setText(prepared))
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def _manage_pronunciation_dictionary(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Dicionário de pronúncia")
        dialog.resize(620, 420)
        layout = QVBoxLayout(dialog)

        help_text = QLabel(
            "Cadastre palavras e expressões ou importe vários termos de arquivos JSON. "
            "A correspondência não diferencia maiúsculas de minúsculas. "
            "A importação combina os arquivos com as entradas atuais; "
            "clique em Salvar para gravar as alterações."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        search = QLineEdit()
        search.setPlaceholderText("Filtrar por palavra ou pronúncia…")
        layout.addWidget(search)

        count_label = QLabel()
        count_label.setObjectName("muted")
        layout.addWidget(count_label)

        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["Palavra ou expressão", "Pronúncia desejada"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        def filter_table(query: str = "") -> None:
            query = query.casefold().strip()
            visible = 0
            for row in range(table.rowCount()):
                term_item = table.item(row, 0)
                pronunciation_item = table.item(row, 1)
                content = " ".join((
                    term_item.text() if term_item else "",
                    pronunciation_item.text() if pronunciation_item else "",
                )).casefold()
                hidden = bool(query and query not in content)
                table.setRowHidden(row, hidden)
                if not hidden:
                    visible += 1
            count_label.setText(
                f"{visible} de {table.rowCount()} entrada(s) exibida(s)."
            )

        def refresh_table(entries: dict[str, str]) -> None:
            ordered = sorted(entries.items(), key=lambda item: item[0].casefold())
            table.setUpdatesEnabled(False)
            table.setRowCount(len(ordered))
            for row, (term, pronunciation) in enumerate(ordered):
                table.setItem(row, 0, QTableWidgetItem(term))
                table.setItem(row, 1, QTableWidgetItem(pronunciation))
            table.setUpdatesEnabled(True)
            filter_table(search.text())

        refresh_table(self.pronunciation_dictionary)
        search.textChanged.connect(filter_table)
        layout.addWidget(table, 1)

        actions = QHBoxLayout()
        import_button = QPushButton("Importar JSON…")
        add_button = QPushButton("Adicionar")
        edit_button = QPushButton("Editar")
        remove_button = QPushButton("Remover")
        actions.addWidget(import_button)
        actions.addWidget(add_button)
        actions.addWidget(edit_button)
        actions.addWidget(remove_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        def current_entries() -> dict[str, str]:
            entries = {}
            for row in range(table.rowCount()):
                term_item = table.item(row, 0)
                pronunciation_item = table.item(row, 1)
                if term_item and pronunciation_item:
                    term = term_item.text().strip()
                    pronunciation = pronunciation_item.text().strip()
                    if term and pronunciation:
                        entries[term] = pronunciation
            return entries

        def import_json_files() -> None:
            filenames, _ = QFileDialog.getOpenFileNames(
                dialog,
                "Importar dicionário de pronúncia",
                str(Path.home()),
                "Arquivos JSON (*.json);;Todos os arquivos (*)",
            )
            if not filenames:
                return

            merged = current_entries()
            added_total = 0
            replaced_total = 0
            skipped_total = 0
            imported_files = 0
            errors = []

            for filename in filenames:
                try:
                    imported, skipped = load_pronunciation_dictionary_json(filename)
                    merged, added, replaced = merge_pronunciation_entries(
                        merged, imported
                    )
                except (OSError, UnicodeError, ValueError) as exc:
                    errors.append(f"{Path(filename).name}: {exc}")
                    continue
                imported_files += 1
                added_total += added
                replaced_total += replaced
                skipped_total += skipped

            if imported_files:
                refresh_table(merged)

            summary = (
                f"Arquivos importados: {imported_files} de {len(filenames)}.\n"
                f"Novas entradas: {added_total}.\n"
                f"Entradas atualizadas: {replaced_total}.\n"
                f"Itens inválidos ignorados: {skipped_total}.\n\n"
                "Clique em Salvar para gravar as alterações."
            )
            if errors:
                summary += "\n\nArquivos não importados:\n" + "\n".join(errors)
            QMessageBox.information(dialog, "Importação do dicionário", summary)
            self.statusBar().showMessage(
                f"Dicionário: {added_total} nova(s) e "
                f"{replaced_total} atualizada(s) via JSON.",
                6000,
            )

        def set_row(row: int, term: str, pronunciation: str) -> None:
            table.setItem(row, 0, QTableWidgetItem(term.strip()))
            table.setItem(row, 1, QTableWidgetItem(pronunciation.strip()))

        def add_entry() -> None:
            term, accepted = QInputDialog.getText(
                dialog, "Nova palavra", "Palavra ou expressão:"
            )
            if not accepted or not term.strip():
                return
            pronunciation, accepted = QInputDialog.getText(
                dialog, "Pronúncia", f"Como pronunciar “{term.strip()}”?"
            )
            if not accepted or not pronunciation.strip():
                return
            for row in range(table.rowCount()):
                existing = table.item(row, 0).text().casefold()
                if existing == term.strip().casefold():
                    set_row(row, term, pronunciation)
                    return
            row = table.rowCount()
            table.insertRow(row)
            set_row(row, term, pronunciation)

        def edit_entry() -> None:
            row = table.currentRow()
            if row < 0:
                return
            current_term = table.item(row, 0).text()
            current_pronunciation = table.item(row, 1).text()
            term, accepted = QInputDialog.getText(
                dialog, "Editar palavra", "Palavra ou expressão:",
                text=current_term,
            )
            if not accepted or not term.strip():
                return
            pronunciation, accepted = QInputDialog.getText(
                dialog, "Editar pronúncia", "Pronúncia desejada:",
                text=current_pronunciation,
            )
            if accepted and pronunciation.strip():
                set_row(row, term, pronunciation)

        def remove_entry() -> None:
            row = table.currentRow()
            if row >= 0:
                table.removeRow(row)

        import_button.clicked.connect(import_json_files)
        add_button.clicked.connect(add_entry)
        edit_button.clicked.connect(edit_entry)
        remove_button.clicked.connect(remove_entry)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        if dialog.exec() == QDialog.DialogCode.Accepted:
            self._save_pronunciation_dictionary(current_entries())


    def _save_advanced_pronunciation_rules(self) -> None:
        self.app_settings.setValue(
            "pronunciation_rules_v1",
            serialize_pronunciation_rules(self.pronunciation_rules),
        )
        self.app_settings.sync()
        self.statusBar().showMessage(
            f"Regras avançadas salvas: {len(self.pronunciation_rules)} regra(s).",
            5000,
        )

    def _edit_advanced_pronunciation_rule(
        self, current: dict | None = None
    ) -> dict | None:
        current = current or {}
        dialog = QDialog(self)
        dialog.setWindowTitle(
            "Editar regra avançada" if current else "Nova regra avançada"
        )
        dialog.resize(520, 360)
        layout = QVBoxLayout(dialog)
        form = QFormLayout()
        term = QLineEdit(str(current.get("term", "")))
        pronunciation = QLineEdit(str(current.get("pronunciation", "")))
        language = QComboBox()
        language.addItem("Todos os idiomas", "all")
        for label, code in (
            ("Português", "portuguese"), ("Inglês", "english"),
            ("Espanhol", "spanish"), ("Francês", "french"),
            ("Italiano", "italian"), ("Alemão", "german"),
            ("Neerlandês", "dutch"),
        ):
            language.addItem(label, code)
        language.setCurrentIndex(max(0, language.findData(current.get("language", "all"))))
        match_mode = QComboBox()
        match_mode.addItem("Palavra inteira", "word")
        match_mode.addItem("Expressão (espaços flexíveis)", "phrase")
        match_mode.addItem("Trecho dentro de palavra", "contains")
        match_mode.setCurrentIndex(
            max(0, match_mode.findData(current.get("match_mode", "word")))
        )
        priority = QLineEdit(str(current.get("priority", 0)))
        priority.setPlaceholderText("0 (maior valor tem prioridade)")
        case_sensitive = QCheckBox("Diferenciar maiúsculas de minúsculas")
        case_sensitive.setChecked(bool(current.get("case_sensitive", False)))
        enabled = QCheckBox("Regra ativa")
        enabled.setChecked(bool(current.get("enabled", True)))
        form.addRow("Texto encontrado", term)
        form.addRow("Substituir pela pronúncia", pronunciation)
        form.addRow("Idioma", language)
        form.addRow("Tipo de correspondência", match_mode)
        form.addRow("Prioridade (-1000 a 1000)", priority)
        form.addRow("", case_sensitive)
        form.addRow("", enabled)
        layout.addLayout(form)
        help_label = QLabel(
            "Regras de maior prioridade são aplicadas primeiro. "
            "A pronúncia substituta não passa novamente pela normalização automática."
        )
        help_label.setWordWrap(True)
        layout.addWidget(help_label)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return None
        if not term.text().strip() or not pronunciation.text().strip():
            QMessageBox.information(
                self, "Regra incompleta",
                "Informe o texto encontrado e a pronúncia substituta.",
            )
            return None
        try:
            priority_value = int(priority.text().strip() or "0")
        except ValueError:
            QMessageBox.information(
                self, "Prioridade inválida", "A prioridade deve ser um número inteiro."
            )
            return None
        if not -1000 <= priority_value <= 1000:
            QMessageBox.information(
                self, "Prioridade inválida", "Use um valor entre -1000 e 1000."
            )
            return None
        return {
            "id": str(current.get("id") or uuid.uuid4().hex),
            "term": term.text().strip(),
            "pronunciation": pronunciation.text().strip(),
            "language": str(language.currentData()),
            "match_mode": str(match_mode.currentData()),
            "case_sensitive": case_sensitive.isChecked(),
            "enabled": enabled.isChecked(),
            "priority": priority_value,
        }

    def _manage_advanced_pronunciation_rules(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Regras avançadas de pronúncia")
        dialog.resize(900, 480)
        layout = QVBoxLayout(dialog)
        intro = QLabel(
            "Crie regras por idioma, tipo de correspondência, diferenciação de caixa "
            "e prioridade. Elas ficam separadas do dicionário simples já existente."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        table = QTableWidget(0, 6)
        table.setHorizontalHeaderLabels(
            ["Texto", "Pronúncia", "Idioma", "Correspondência", "Prioridade", "Ativa"]
        )
        table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        rules = [dict(rule) for rule in self.pronunciation_rules]
        language_labels = {
            "all": "Todos", "portuguese": "Português", "english": "Inglês",
            "spanish": "Espanhol", "french": "Francês", "italian": "Italiano",
            "german": "Alemão", "dutch": "Neerlandês",
        }
        mode_labels = {
            "word": "Palavra inteira", "phrase": "Expressão",
            "contains": "Trecho",
        }

        def refresh(selected: int = 0) -> None:
            table.setRowCount(0)
            ordered = sorted(
                enumerate(rules),
                key=lambda item: (
                    -int(item[1].get("priority", 0)),
                    -len(str(item[1].get("term", ""))),
                ),
            )
            for index, rule in ordered:
                row = table.rowCount()
                table.insertRow(row)
                values = (
                    rule["term"], rule["pronunciation"],
                    language_labels.get(rule["language"], rule["language"]),
                    mode_labels.get(rule["match_mode"], rule["match_mode"]),
                    str(rule["priority"]),
                    "Sim" if rule["enabled"] else "Não",
                )
                for column, value in enumerate(values):
                    cell = QTableWidgetItem(str(value))
                    cell.setData(Qt.ItemDataRole.UserRole, index)
                    table.setItem(row, column, cell)
            if table.rowCount():
                table.selectRow(min(max(selected, 0), table.rowCount() - 1))

        layout.addWidget(table, 1)
        actions = QHBoxLayout()
        add_button = QPushButton("Adicionar")
        edit_button = QPushButton("Editar")
        remove_button = QPushButton("Remover")
        actions.addWidget(add_button)
        actions.addWidget(edit_button)
        actions.addWidget(remove_button)
        actions.addStretch(1)
        layout.addLayout(actions)

        def selected_index() -> int:
            row = table.currentRow()
            if row < 0 or table.item(row, 0) is None:
                return -1
            return int(table.item(row, 0).data(Qt.ItemDataRole.UserRole))

        def add_rule() -> None:
            rule = self._edit_advanced_pronunciation_rule()
            if rule is not None:
                rules.append(rule)
                refresh(len(rules) - 1)

        def edit_rule() -> None:
            index = selected_index()
            if index < 0:
                return
            rule = self._edit_advanced_pronunciation_rule(rules[index])
            if rule is not None:
                rules[index] = rule
                refresh(index)

        def remove_rule() -> None:
            index = selected_index()
            if index >= 0:
                rules.pop(index)
                refresh(min(index, len(rules) - 1))

        add_button.clicked.connect(add_rule)
        edit_button.clicked.connect(edit_rule)
        remove_button.clicked.connect(remove_rule)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        refresh()
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.pronunciation_rules = rules
            self._save_advanced_pronunciation_rules()

    def _save_tts_output_settings(self, *_args) -> None:
        if not hasattr(self, "tts_output_directory"):
            return
        self.app_settings.setValue(
            "tts_output_directory", self.tts_output_directory.text().strip()
        )
        self.app_settings.setValue(
            "tts_auto_filename", self.tts_auto_filename.isChecked()
        )

    def _choose_tts_output_directory(self) -> None:
        current = self.tts_output_directory.text().strip() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(
            self,
            "Escolher pasta para salvar os áudios",
            current,
        )
        if directory:
            self.tts_output_directory.setText(directory)
            self._save_tts_output_settings()

    def _choose_voice_library_directory(self) -> None:
        current = self.voice_library_directory.text().strip() or str(Path.home())
        directory = QFileDialog.getExistingDirectory(
            self, "Escolher pasta principal da biblioteca de vozes", current
        )
        if directory:
            self.voice_library_directory.setText(directory)
            self._save_text_setting("voice_library_directory", directory)

    def _choose_voice(self) -> None:
        root_text = self.voice_library_directory.text().strip()
        if not root_text:
            self._choose_voice_library_directory()
            root_text = self.voice_library_directory.text().strip()
        root = Path(root_text).expanduser() if root_text else None
        if root is None or not root.is_dir():
            QMessageBox.information(
                self, "Biblioteca não definida",
                "Escolha uma pasta principal válida para procurar as vozes."
            )
            return
        current_text = self.voice_path.text().strip()
        previous_path = Path(current_text).expanduser() if current_text else None
        dialog = VoiceLibraryDialog(root, self.player, previous_path, self)
        if dialog.exec() == QDialog.DialogCode.Accepted and dialog.selected_path:
            self.voice_path.setText(str(dialog.selected_path))

    def _preview_selected_voice(self) -> None:
        path = Path(self.voice_path.text().strip()).expanduser()
        if not path.is_file():
            QMessageBox.information(self, "Voz não selecionada", "Selecione uma voz primeiro.")
            return
        if path.suffix.lower() == ".safetensors":
            QMessageBox.information(
                self, "Estado de voz",
                "O .safetensors contém o estado processado da voz, não áudio reproduzível. "
                "Para ouvir a referência, selecione o WAV/MP3 original."
            )
            return
        source = QUrl.fromLocalFile(str(path.resolve()))
        action = decide_playback_action(
            self.player.source().toLocalFile(),
            path,
            is_playing=(
                self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            ),
            is_paused=(
                self.player.playbackState() == QMediaPlayer.PlaybackState.PausedState
            ),
        )
        if action == "stop":
            self.player.stop()
            return
        if action == "resume":
            self.player.play()
            return
        self.player.setSource(source)
        self.player.play()

    def _prepare_voice_state(self) -> None:
        source = Path(self.voice_path.text().strip()).expanduser()
        if not source.is_file():
            QMessageBox.information(
                self,
                "Selecione uma voz de referência",
                "Escolha primeiro um arquivo de áudio de referência. "
                "Arquivos .safetensors já são estados de voz preparados.",
            )
            return
        if source.suffix.lower() == ".safetensors":
            QMessageBox.information(
                self,
                "Voz já preparada",
                "Este arquivo já é um estado de voz .safetensors. "
                "Ele será carregado diretamente pelo Pocket TTS.",
            )
            return

        output, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar estado reutilizável da voz",
            str(source.with_suffix(".safetensors")),
            "Estado de voz Pocket TTS (*.safetensors)",
        )
        if not output:
            return
        if not output.lower().endswith(".safetensors"):
            output += ".safetensors"
        destination = Path(output).expanduser()
        language = str(self.tts_language.currentData())

        def export() -> Path:
            return self.tts.export_voice_state(source, destination, language)

        def completed(result: object) -> None:
            saved = Path(result)
            self.voice_path.setText(str(saved))
            QMessageBox.information(
                self,
                "Estado de voz preparado",
                "O estado foi salvo e selecionado como voz padrão:\\n"
                f"{saved}\\n\\n"
                "Nas próximas gerações, o Pocket TTS carregará esse estado sem "
                "reprocessar o áudio original.",
            )

        self._run_task("Preparando estado reutilizável da voz…", export, completed)

    def _choose_audio(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar áudio para transcrever",
            str(Path.home()),
            "Áudio (*.wav *.mp3 *.flac *.m4a *.ogg *.aac);;Todos os arquivos (*)",
        )
        if path:
            self.audio_path.setText(path)
            self.transcript.clear()
            self.transcript_segments = []
            self.save_srt_button.setEnabled(False)

    @staticmethod
    def _xfce_hotkey_property() -> str:
        return "/commands/custom/<Primary><Shift>space"

    @staticmethod
    def _xfce_hotkey_command() -> str:
        run_script = Path(__file__).resolve().parents[2] / "run.sh"
        return f'/usr/bin/bash "{run_script}" --clipboard-audio'

    def _xfce_hotkey_is_registered(self) -> bool:
        if not shutil.which("xfconf-query"):
            return False
        try:
            result = subprocess.run(
                ["xfconf-query", "--channel", "xfce4-keyboard-shortcuts",
                 "--property", self._xfce_hotkey_property()],
                capture_output=True, text=True, timeout=2, check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        return result.returncode == 0 and result.stdout.strip() == self._xfce_hotkey_command()

    def _stop_global_hotkey(self) -> None:
        self.hotkey_monitor_timer.stop()
        listener, self.hotkey_listener = self.hotkey_listener, None
        if listener is not None:
            try:
                listener.stop()
                listener.join(timeout=0.5)
            except Exception:  # noqa: BLE001
                pass

    def _restart_global_hotkey(self) -> None:
        self._stop_global_hotkey()
        self._start_global_hotkey()

    def _check_global_hotkey(self) -> None:
        listener = self.hotkey_listener
        if listener is None or listener.is_alive():
            return
        detail = "o listener encerrou inesperadamente"
        try:
            listener.join(timeout=0)
        except Exception as exc:  # noqa: BLE001
            detail = str(exc).splitlines()[0] or detail
        self.hotkey_listener = None
        self.hotkey_monitor_timer.stop()
        self.statusBar().showMessage(
            f"Atalho global parou: {detail}. Use Arquivo → Reativar atalho global.",
            15000,
        )

    def _handle_local_command(self, command: str) -> None:
        if command == "clipboard-audio":
            self._generate_audio_from_clipboard()
        elif command == "activate":
            if self.isMinimized():
                self.showNormal()
            self.raise_()
            self.activateWindow()

    def _start_global_hotkey(self) -> None:
        # Prefer the Xfce shortcut manager; it works when Joe is not focused.
        if self._xfce_hotkey_is_registered():
            self._stop_global_hotkey()
            self.statusBar().showMessage(
                "Atalho global do Xfce ativo: Ctrl+Shift+Espaço", 8000
            )
            return
        try:
            from pynput import keyboard
            listener = keyboard.GlobalHotKeys(
                {"<ctrl>+<shift>+<space>": lambda: self.hotkey_bridge.activated.emit()}
            )
            listener.start()
            self.hotkey_listener = listener
            self.hotkey_monitor_timer.start()
            QTimer.singleShot(700, self._check_global_hotkey)
        except Exception as exc:  # noqa: BLE001
            self.hotkey_listener = None
            self.hotkey_monitor_timer.stop()
            session_type = os.environ.get("XDG_SESSION_TYPE", "desconhecido")
            if session_type == "wayland" or os.environ.get("WAYLAND_DISPLAY"):
                reason = "Wayland limita a captura global pelo pynput; use o atalho nativo do desktop."
            elif not os.environ.get("DISPLAY"):
                reason = "DISPLAY não está definido; inicie o Joe pela sessão gráfica."
            else:
                reason = str(exc).splitlines()[0]
            self.statusBar().showMessage(
                f"Atalho global indisponível: {reason}. Use Arquivo → Gerar áudio do texto copiado.",
                15000,
            )

    @staticmethod
    def _remove_clipboard_temp_dir(directory: Path | None) -> None:
        if directory is not None:
            shutil.rmtree(directory, ignore_errors=True)

    def _clipboard_dialog_closed(self, dialog: ClipboardAudioDialog) -> None:
        if self.clipboard_audio_window is dialog:
            self.clipboard_audio_window = None
        worker = self.clipboard_generation_worker
        if worker is not None:
            worker.cancel()
        else:
            self._remove_clipboard_temp_dir(dialog.temp_dir)
            if self.clipboard_temp_dir == dialog.temp_dir:
                self.clipboard_temp_dir = None
            dialog.temp_dir = None
        if self.clipboard_only_mode and worker is None:
            QTimer.singleShot(0, QApplication.quit)

    def _generate_audio_from_clipboard(self) -> None:
        shortcut_action = decide_clipboard_shortcut_action(
            dialog_open=self.clipboard_audio_window is not None,
            generation_active=self.clipboard_generation_worker is not None,
        )
        if shortcut_action == "focus":
            dialog = self.clipboard_audio_window
            if dialog is not None:
                if dialog.isMinimized():
                    dialog.showNormal()
                elif not dialog.isVisible():
                    dialog.show()
                dialog.raise_()
                dialog.activateWindow()
            message = (
                "A geração do texto copiado já está em andamento."
                if self.clipboard_generation_worker is not None
                else "A janela de áudio rápido já está aberta. "
                "Use-a para reproduzir ou salvar o áudio gerado."
            )
            self.statusBar().showMessage(message, 5000)
            return

        if shortcut_action == "busy":
            self.statusBar().showMessage(
                "A geração do texto copiado já está em andamento.", 4000
            )
            return

        if self.active_worker is not None:
            self.statusBar().showMessage(
                "Aguarde a tarefa atual terminar antes de gerar outro áudio.", 5000
            )
            if self.clipboard_only_mode:
                QTimer.singleShot(0, QApplication.quit)
            return

        copied_text = QApplication.clipboard().text().strip()
        if not copied_text:
            self.statusBar().showMessage(
                "O clipboard está vazio. Copie um texto e pressione Ctrl+Shift+Espaço.",
                6000,
            )
            if self.clipboard_only_mode:
                QTimer.singleShot(0, QApplication.quit)
            return

        language = str(self.tts_language.currentData())
        try:
            speech_text = self._prepared_speech_text(copied_text, language)
            remote = (
                self._colab_client()
                if self.processing_mode.currentData() == "colab"
                else None
            )
            chunk_size = int(self.clipboard_chunk_size.currentData())
            chunks = split_text_into_chunks(speech_text, max_chars=chunk_size)
            if not chunks:
                raise ValueError("Não há texto para converter em áudio.")
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Não foi possível iniciar o TTS: {exc}", 8000)
            if self.clipboard_only_mode:
                QTimer.singleShot(0, QApplication.quit)
            return

        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="joe-clipboard-audio-"))
        except OSError as exc:
            self.statusBar().showMessage(
                f"Não foi possível criar áudio temporário: {exc}", 8000
            )
            if self.clipboard_only_mode:
                QTimer.singleShot(0, QApplication.quit)
            return

        destination = temp_dir / "clipboard_audio.wav"
        voice = self.voice_path.text().strip() or None

        if remote is not None:
            def generate_clipboard_chunk(chunk_text: str, output_path: Path) -> Path:
                audio_data = remote.generate_audio(chunk_text, language, voice)
                write_bytes_atomically(output_path, audio_data)
                return output_path
        else:
            def generate_clipboard_chunk(chunk_text: str, output_path: Path) -> Path:
                return self.tts.generate(
                    chunk_text, language, str(output_path), voice
                )

        dialog = ClipboardAudioDialog(
            self.player,
            self.tts_output_directory.text().strip(),
            self,
        )
        dialog.temp_dir = temp_dir
        dialog.set_text_preview(copied_text)
        dialog.start_measurement(remote=remote is not None)
        dialog.closed.connect(lambda: self._clipboard_dialog_closed(dialog))
        dialog.saved.connect(
            lambda output_path, performance: self._record_clipboard_history(
                output_path,
                performance,
                copied_text,
                language,
                voice,
                remote is not None,
            )
        )
        self.clipboard_audio_window = dialog
        self.clipboard_temp_dir = temp_dir
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

        worker = ClipboardChunkWorker(
            chunks,
            generate_clipboard_chunk,
            temp_dir,
            destination,
            remote=remote is not None,
        )
        self.clipboard_generation_worker = worker

        def chunk_ready(index: int, path: str, total: int) -> None:
            if not dialog.is_closing:
                dialog.add_chunk(index, Path(path), total)

        def progress(message: str) -> None:
            if not dialog.is_closing:
                dialog.set_progress(message)

        def cleanup_closed_dialog() -> None:
            self._remove_clipboard_temp_dir(temp_dir)
            dialog.temp_dir = None
            if self.clipboard_temp_dir == temp_dir:
                self.clipboard_temp_dir = None
            if self.clipboard_only_mode:
                QTimer.singleShot(0, QApplication.quit)

        def finished(result: object) -> None:
            self.clipboard_generation_worker = None
            if dialog.is_closing:
                cleanup_closed_dialog()
                return
            dialog.set_ready(
                result if isinstance(result, dict) else {"path": str(result)}
            )
            self.statusBar().showMessage("Áudio do clipboard gerado e unido.", 5000)

        def failed(message: str) -> None:
            self.clipboard_generation_worker = None
            detail = message.split("\n\n", 1)[0].strip() or "Erro desconhecido"
            if dialog.is_closing:
                cleanup_closed_dialog()
            else:
                dialog.set_failed(detail)
                self.statusBar().showMessage(
                    f"Falha ao gerar áudio do clipboard: {detail}", 8000
                )

        def cancelled() -> None:
            self.clipboard_generation_worker = None
            cleanup_closed_dialog()

        worker.signals.chunk_ready.connect(chunk_ready)
        worker.signals.progress.connect(progress)
        worker.signals.finished.connect(finished)
        worker.signals.failed.connect(failed)
        worker.signals.cancelled.connect(cancelled)
        self.thread_pool.start(worker)

    def _generate_audio(self) -> None:
        self._enqueue_current_generation()

    def _enqueue_current_generation(self) -> None:
        text = self.tts_text.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Texto vazio", "Digite o texto que deseja converter.")
            return

        output_directory_text = self.tts_output_directory.text().strip()
        if not output_directory_text:
            QMessageBox.information(
                self, "Pasta não definida", "Escolha uma pasta para salvar os áudios."
            )
            return
        output_directory = Path(output_directory_text).expanduser()

        if self.tts_auto_filename.isChecked():
            try:
                destination = build_audio_output_path(output_directory)
                reserved = {
                    str(Path(task.output_path).expanduser().resolve(strict=False))
                    for task in self.generation_queue.tasks()
                    if task.status not in self.generation_queue.FINISHED_STATES
                }
                while str(destination.resolve(strict=False)) in reserved:
                    destination = destination.with_name(
                        f"{destination.stem}_001{destination.suffix}"
                    )
            except OSError as exc:
                QMessageBox.critical(
                    self, "Erro na pasta de saída", f"Não foi possível preparar a pasta:\\n{exc}"
                )
                return
        else:
            output, _ = QFileDialog.getSaveFileName(
                self,
                "Salvar áudio gerado",
                str(output_directory / "joe-audio.wav"),
                "Áudio WAV (*.wav)",
            )
            if not output:
                return
            if not output.lower().endswith(".wav"):
                output += ".wav"
            destination = Path(output).expanduser()
            reserved = {
                str(Path(task.output_path).expanduser().resolve(strict=False))
                for task in self.generation_queue.tasks()
                if task.status not in self.generation_queue.FINISHED_STATES
            }
            if str(destination.resolve(strict=False)) in reserved:
                QMessageBox.information(
                    self,
                    "Arquivo já reservado",
                    "Outra tarefa da fila já usa esse caminho. Escolha outro nome para evitar "
                    "que uma geração sobrescreva a outra.",
                )
                return
            self.tts_output_directory.setText(str(destination.parent))
            self._save_tts_output_settings()
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                QMessageBox.critical(
                    self, "Erro na pasta de saída", f"Não foi possível criar a pasta:\\n{exc}"
                )
                return

        language = str(self.tts_language.currentData())
        try:
            speech_text = self._prepared_speech_text(text, language)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Erro na normalização", str(exc))
            return

        remote = self.processing_mode.currentData() == "colab"
        remote_url = self.colab_url.text().strip() if remote else ""
        remote_token = self.colab_token.text().strip() if remote else ""
        if remote:
            try:
                self._colab_client()
            except ValueError as exc:
                QMessageBox.information(self, "Configure a API do Colab", str(exc))
                return

        try:
            selected_voice = self.voice_path.text().strip() or None
            selected_voice_key = canonical_voice_path(selected_voice) if selected_voice else ""
            selected_voice_alias = self.favorite_voices.get(
                selected_voice_key,
                Path(selected_voice).stem if selected_voice else "Voz padrão",
            )
            task = self.generation_queue.enqueue(
                text=text,
                speech_text=speech_text,
                language=language,
                output_path=str(destination),
                voice_path=selected_voice,
                voice_alias=selected_voice_alias,
                remote=remote,
                remote_url=remote_url,
                remote_token=remote_token,
            )
        except ValueError as exc:
            QMessageBox.information(self, "Tarefa inválida", str(exc))
            return

        self.statusBar().showMessage(f"Tarefa adicionada à fila: {task.output_name}", 4000)
        self._refresh_generation_queue()
        self._start_next_generation()

    def _open_generation_queue(self) -> None:
        if self.generation_queue_dialog is None:
            self.generation_queue_dialog = GenerationQueueDialog(
                self.generation_queue,
                self._enqueue_current_generation,
                self._cancel_generation_task,
                self._retry_generation_task,
                self._prioritize_generation_task,
                self,
            )
        self.generation_queue_dialog.refresh()
        self.generation_queue_dialog.show()
        self.generation_queue_dialog.raise_()
        self.generation_queue_dialog.activateWindow()

    def _refresh_generation_queue(self) -> None:
        active_count = sum(
            task.status in {"queued", "running", "cancelling"}
            for task in self.generation_queue.tasks()
        )
        self.tts_queue_button.setText(f"Fila de geração ({active_count})")
        if self.generation_queue_dialog is not None:
            self.generation_queue_dialog.refresh()

    def _start_next_generation(self) -> None:
        if self.active_generation_worker is not None:
            return
        task = self.generation_queue.next_pending()
        if task is None:
            self._refresh_generation_queue()
            return

        self.generation_queue.start(task.task_id)
        worker = TTSGenerationWorker(task, self.tts)
        self.active_generation_worker = worker
        worker.signals.progress.connect(self._generation_progress)
        worker.signals.finished.connect(self._generation_completed)
        worker.signals.failed.connect(self._generation_failed)
        worker.signals.cancelled.connect(self._generation_cancelled)

        self.tts_generation_started_at = time.monotonic()
        self.tts_speed_info.setText(
            "Medição do Joe neste computador — "
            f"{format_elapsed_status(0)}"
        )
        self.tts_elapsed_timer.start()
        backend = "Google Colab" if task.remote else "Pocket TTS local"
        self._set_busy(True, f"Gerando {task.output_name} com {backend}…")
        self._refresh_generation_queue()
        self.thread_pool.start(worker)

    def _generation_progress(self, task_id: str, message: str) -> None:
        self.generation_queue.update_progress(task_id, message)
        self.statusBar().showMessage(message)
        self._refresh_generation_queue()

    def _generation_completed(self, task_id: str, result: object) -> None:
        task = self.generation_queue.get(task_id)
        self.generation_queue.complete(task_id)
        if task is not None:
            self._record_task_history(task, result)
        self.active_generation_worker = None
        self.tts_elapsed_timer.stop()
        self.tts_generation_started_at = None
        self._set_busy(False, "Tarefa concluída")
        self._audio_generated(result)
        self._refresh_generation_queue()
        self._start_next_generation()

    def _generation_failed(self, task_id: str, message: str) -> None:
        self.generation_queue.fail(task_id, message)
        self.active_generation_worker = None
        self.tts_elapsed_timer.stop()
        self.tts_generation_started_at = None
        self.tts_speed_info.setText("Geração falhou. Veja os detalhes na fila para repetir.")
        self._set_busy(False, "Falha na tarefa; continuando a fila")
        failed_task = self.generation_queue.get(task_id)
        task_name = failed_task.output_name if failed_task is not None else task_id
        error_summary = message.splitlines()[0] if message.splitlines() else "Erro desconhecido"
        self.statusBar().showMessage(
            f"Falha em {task_name}: {error_summary}",
            8000,
        )
        self._refresh_generation_queue()
        self._start_next_generation()

    def _generation_cancelled(self, task_id: str) -> None:
        self.generation_queue.mark_cancelled(task_id)
        self.active_generation_worker = None
        self.tts_elapsed_timer.stop()
        self.tts_generation_started_at = None
        self.tts_speed_info.setText("Geração cancelada.")
        self._set_busy(False, "Tarefa cancelada; continuando a fila")
        self._refresh_generation_queue()
        self._start_next_generation()

    def _cancel_generation_task(self, task_id: str) -> None:
        task = self.generation_queue.get(task_id)
        if task is None:
            return
        previous_status = task.status
        self.generation_queue.request_cancel(task_id)
        if previous_status == "running" and self.active_generation_worker is not None:
            self.active_generation_worker.cancel()
            self.statusBar().showMessage(
                "Cancelamento solicitado. Um bloco local já iniciado precisa terminar; "
                "no Colab, o resultado remoto será descartado.",
                9000,
            )
        else:
            self._start_next_generation()
        self._refresh_generation_queue()

    def _retry_generation_task(self, task_id: str) -> None:
        try:
            self.generation_queue.retry(task_id)
        except (KeyError, ValueError, RuntimeError) as exc:
            self.statusBar().showMessage(str(exc), 5000)
            return
        self._refresh_generation_queue()
        self._start_next_generation()

    def _prioritize_generation_task(self, task_id: str) -> None:
        try:
            changed = self.generation_queue.prioritize(task_id)
        except KeyError:
            return
        if changed:
            self.statusBar().showMessage(
                "Tarefa marcada como urgente; será a próxima após a geração atual.",
                5000,
            )
        self._refresh_generation_queue()

    def _update_tts_elapsed(self) -> None:
        if self.tts_generation_started_at is None:
            self.tts_elapsed_timer.stop()
            return
        elapsed = time.monotonic() - self.tts_generation_started_at
        self.tts_speed_info.setText(
            "Medição do Joe neste computador — "
            f"{format_elapsed_status(elapsed)}"
        )

    def _audio_generated(self, result: object) -> None:
        self.tts_elapsed_timer.stop()
        self.tts_generation_started_at = None

        metrics = result if isinstance(result, dict) else {}
        self.generated_audio = Path(metrics.get("path", result))
        self.tts_play_button.setEnabled(True)
        self.tts_info.setText(f"Áudio salvo em: {self.generated_audio}")

        performance = metrics.get("performance")
        if isinstance(performance, AudioPerformance):
            self.tts_speed_info.setText(format_performance_report(performance))
        else:
            self.tts_speed_info.setText(
                "Áudio gerado, mas não foi possível calcular as métricas de desempenho."
            )

    def _play_generated_audio(self) -> None:
        if not self.generated_audio or not self.generated_audio.is_file():
            return
        source = QUrl.fromLocalFile(str(self.generated_audio.resolve()))
        action = decide_playback_action(
            self.player.source().toLocalFile(),
            self.generated_audio,
            is_playing=(
                self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
            ),
            is_paused=(
                self.player.playbackState() == QMediaPlayer.PlaybackState.PausedState
            ),
        )
        if action == "stop":
            self.player.stop()
            return
        if action == "resume":
            self.player.play()
            return
        self.player.setSource(source)
        self.player.play()

    def _transcribe_audio(self) -> None:
        source = self.audio_path.text().strip()
        if not source:
            QMessageBox.information(self, "Áudio não selecionado", "Selecione um arquivo de áudio.")
            return

        self.transcript_audio_path = str(Path(source).expanduser().resolve())
        self.transcript_segments = []
        self.transcript_sync_list.clear()
        self.transcript_sync_button.setEnabled(False)
        self.transcript_sync_list.setEnabled(False)
        self.transcript_sync_info.setText("Aguardando os tempos da nova transcrição…")
        want_timestamps = self.timestamps_check.isChecked()
        if self.processing_mode.currentData() == "colab":
            try:
                remote = self._colab_client()
            except ValueError as exc:
                QMessageBox.information(self, "Configure a API do Colab", str(exc))
                return
            self._run_task(
                "Enviando áudio para o Google Colab e transcrevendo…",
                lambda: remote.transcribe(source, want_timestamps),
                self._transcription_finished,
            )
        else:
            device_preference = str(self.asr_device.currentData())
            self._run_task(
                "Carregando Parakeet Redux e transcrevendo localmente…",
                lambda: self.asr.transcribe(source, want_timestamps, device_preference),
                self._transcription_finished,
            )

    def _transcription_finished(self, result: object) -> None:
        data = result if isinstance(result, dict) else {}
        self.transcript_segments = normalize_timed_segments(
            list(data.get("segments", []))
        )
        self.transcript_sync_list.clear()
        for index, segment in enumerate(self.transcript_segments, start=1):
            start = float(segment["start"])
            end = float(segment["end"])
            start_label = self._format_transcript_time(start)
            end_label = self._format_transcript_time(end)
            item = QListWidgetItem(
                f"{index:03d}  [{start_label} – {end_label}]  {segment['text']}"
            )
            item.setData(Qt.ItemDataRole.UserRole, index - 1)
            self.transcript_sync_list.addItem(item)
        self.transcript_sync_button.setEnabled(bool(self.transcript_segments))
        self.transcript_sync_list.setEnabled(bool(self.transcript_segments))
        self.transcript_sync_info.setText(
            f"{len(self.transcript_segments)} trecho(s) sincronizado(s)"
            if self.transcript_segments else
            "Este backend não retornou tempos de segmento; a sincronização não está disponível."
        )
        if self.transcript_segments:
            visible_text = "\n".join(
                str(segment.get("text", "")).strip()
                for segment in self.transcript_segments
            )
        else:
            visible_text = str(data.get("text", ""))
        self.transcript.setPlainText(visible_text)
        self.transcription_backend = str(data.get("backend", "Parakeet Redux"))
        note = str(data.get("note", ""))
        if data.get("device"):
            note = (note + f" Dispositivo: {data.get('device')}.").strip()
        if self.timestamps_check.isChecked() and not self.transcript_segments:
            note += " Os tempos de segmento não foram retornados por este backend."
        self.save_srt_button.setEnabled(bool(self.transcript_segments))
        self.asr_info.setText(f"{self.transcription_backend} — {note}".strip(" —"))

    @staticmethod
    def _format_transcript_time(seconds: float) -> str:
        total = max(0, int(round(seconds * 1000)))
        minutes, remainder = divmod(total, 60_000)
        whole_seconds, milliseconds = divmod(remainder, 1000)
        return f"{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"

    def _refresh_transcript_sync_labels(self) -> None:
        if not self.transcript_segments or not hasattr(self, "transcript_sync_list"):
            return
        edited_lines = [
            line.strip() for line in self.transcript.toPlainText().splitlines()
            if line.strip()
        ]
        if len(edited_lines) != len(self.transcript_segments):
            self.transcript_sync_info.setText(
                "Para manter a sincronização, mantenha uma linha para cada trecho com tempo."
            )
            return
        for index, (segment, text) in enumerate(
            zip(self.transcript_segments, edited_lines)
        ):
            start = self._format_transcript_time(float(segment["start"]))
            end = self._format_transcript_time(float(segment["end"]))
            item = self.transcript_sync_list.item(index)
            if item is not None:
                item.setText(f"{index + 1:03d}  [{start} – {end}]  {text}")
        self.transcript_sync_info.setText(
            f"{len(self.transcript_segments)} trecho(s) sincronizado(s)"
        )

    def _on_transcript_audio_path_changed(self, value: str) -> None:
        selected = str(Path(value).expanduser().resolve()) if value.strip() else ""
        if selected and self.transcript_audio_path and selected != self.transcript_audio_path:
            self.transcript_sync_button.setEnabled(False)
            self.transcript_sync_list.setEnabled(False)
            self.transcript_sync_info.setText(
                "O arquivo selecionado mudou. Faça uma nova transcrição para sincronizar."
            )

    def _seek_transcript_segment(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(index, int) or not 0 <= index < len(self.transcript_segments):
            return
        segment = self.transcript_segments[index]
        source = self.audio_path.text().strip()
        if not source or not Path(source).is_file():
            return
        target = Path(source).resolve()
        current = self.player.source().toLocalFile()
        if not current or Path(current).resolve() != target:
            self.player.stop()
            self.player.setSource(QUrl.fromLocalFile(str(target)))
        self.player.setPosition(int(round(segment["start"] * 1000)))
        self.statusBar().showMessage(
            f"Posicionado no trecho {index + 1} da transcrição.", 3000
        )

    def _play_transcript_with_sync(self) -> None:
        source = self.audio_path.text().strip()
        if not source or not Path(source).is_file():
            QMessageBox.information(
                self, "Áudio indisponível",
                "Selecione novamente o arquivo de áudio usado nesta transcrição."
            )
            return
        if not self.transcript_segments:
            return
        target = Path(source).resolve()
        if self.transcript_audio_path and target != Path(self.transcript_audio_path):
            QMessageBox.information(
                self, "Transcrição de outro arquivo",
                "O áudio selecionado não corresponde à transcrição atual. "
                "Faça uma nova transcrição antes de usar a sincronização."
            )
            return
        current = self.player.source().toLocalFile()
        if not current or Path(current).resolve() != target:
            self.player.stop()
            self.player.setSource(QUrl.fromLocalFile(str(target)))
        self.player.setPosition(int(round(self.transcript_segments[0]["start"] * 1000)))
        self.player.play()
        self.statusBar().showMessage("Reprodução sincronizada com a transcrição.")

    def _update_transcript_sync_position(self, position_ms: int) -> None:
        if not self.transcript_segments or not hasattr(self, "transcript_sync_list"):
            return
        index = active_segment_index(self.transcript_segments, position_ms)
        if index is None:
            if self.transcript_sync_list.currentRow() >= 0:
                self.transcript_sync_list.setCurrentRow(-1)
            return
        if self.transcript_sync_list.currentRow() != index:
            self.transcript_sync_list.setCurrentRow(index)
            self.transcript_sync_list.scrollToItem(
                self.transcript_sync_list.item(index)
            )

    def _copy_transcript(self) -> None:
        text = self.transcript.toPlainText()
        if not text.strip():
            return
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage("Transcrição copiada.", 3000)

    def _save_transcript(self) -> None:
        text = self.transcript.toPlainText()
        if not text.strip():
            QMessageBox.information(self, "Sem transcrição", "Não há texto para salvar.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar transcrição",
            str(Path.home() / "transcricao.txt"),
            "Arquivo de texto (*.txt)",
        )
        if not path:
            return
        if not path.lower().endswith(".txt"):
            path += ".txt"
        try:
            Path(path).write_text(text, encoding="utf-8")
        except OSError as exc:
            QMessageBox.critical(self, "Erro ao salvar", str(exc))
            return
        self.statusBar().showMessage(f"Transcrição salva: {path}", 5000)

    def _save_srt(self) -> None:
        if not self.transcript_segments:
            QMessageBox.information(
                self,
                "Tempos indisponíveis",
                "Este backend não retornou segmentos com marcações de tempo.",
            )
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar legendas",
            str(Path.home() / "transcricao.srt"),
            "Legendas SubRip (*.srt)",
        )
        if not path:
            return
        if not path.lower().endswith(".srt"):
            path += ".srt"
        edited_lines = [
            line.strip()
            for line in self.transcript.toPlainText().splitlines()
            if line.strip()
        ]
        if len(edited_lines) != len(self.transcript_segments):
            QMessageBox.information(
                self,
                "Revise os segmentos",
                "Para exportar o texto editado, mantenha uma linha para cada segmento "
                "com tempo. Não adicione nem remova linhas.",
            )
            return
        edited_segments = [
            {**segment, "text": edited_lines[index]}
            for index, segment in enumerate(self.transcript_segments)
        ]
        try:
            Path(path).write_text(build_srt(edited_segments), encoding="utf-8")
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "Erro ao exportar SRT", str(exc))
            return
        self.statusBar().showMessage(f"Legendas salvas: {path}", 5000)

    def _run_task(
        self,
        status: str,
        function: Callable[[], object],
        on_success: Callable[[object], None],
    ) -> None:
        self._set_busy(True, status)
        worker = FunctionWorker(function)
        self.active_worker = worker

        def finished(result: object) -> None:
            self.active_worker = None
            self._set_busy(False, "Concluído")
            on_success(result)

        def failed(message: str) -> None:
            self.active_worker = None
            if self.tts_generation_started_at is not None:
                self.tts_elapsed_timer.stop()
                self.tts_generation_started_at = None
                self.tts_speed_info.setText(
                    "Geração interrompida. Consulte os detalhes do erro."
                )
            self._set_busy(False, "Falha")
            QMessageBox.critical(self, "Falha no processamento", message)

        worker.signals.finished.connect(finished)
        worker.signals.failed.connect(failed)
        self.thread_pool.start(worker)

    def _set_busy(self, busy: bool, message: str) -> None:
        for widget in (
            self.tts_generate_button,
            self.tts_play_button,
            self.tts_favorites_button,
            self.tts_settings_button,
            self.tts_language,
            self.clipboard_chunk_size,
            self.voice_path,
            self.voice_button,
            self.voice_preview_button,
            self.voice_library_directory,
            self.voice_library_button,
            self.voice_prepare_button,
            self.favorite_voice_alias,
            self.favorite_voice_save_button,
            self.favorite_voice_remove_button,
            self.tts_output_directory,
            self.tts_browse_output_button,
            self.tts_auto_filename,
            self.tts_normalization_enabled,
            self.tts_dictionary_button,
            self.tts_preview_button,
            self.audio_choose_button,
            self.asr_button,
            self.asr_device,
            self.timestamps_check,
            self.colab_test_button,
        ):
            widget.setEnabled(not busy)
        if busy:
            self.tts_play_button.setEnabled(False)
        elif self.generated_audio and self.generated_audio.is_file():
            self.tts_play_button.setEnabled(True)
        self._update_mode_widgets()
        self.status_label.setText(message)
        self.statusBar().showMessage(message)

    def _show_about(self) -> None:
        QMessageBox.about(
            self,
            "Sobre o Joe",
            "<b>Joe — Audio Studio 0.1.0</b><br><br>"
            "Aplicativo local para texto em áudio e áudio em texto.<br>"
            "TTS: Kyutai Pocket TTS<br>"
            "STT: Moondream Parakeet Redux<br><br>"
            "Os modelos são executados localmente após o download inicial.",
        )

    def closeEvent(self, event) -> None:
        pending_tasks = self.generation_queue.pending_count()
        if pending_tasks and not self.clipboard_only_mode:
            answer = QMessageBox.question(
                self,
                "Tarefas aguardando",
                f"Há {pending_tasks} tarefa(s) aguardando na fila. Fechar o Joe descartará "
                "essas tarefas que ainda não começaram. Deseja continuar?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
        self.generation_queue.cancel_pending()
        if self.active_generation_worker is not None:
            active_task = self.generation_queue.active_task()
            if active_task is not None:
                self.generation_queue.request_cancel(active_task.task_id)
            self.active_generation_worker.cancel()
        if self.clipboard_generation_worker is not None:
            self.clipboard_generation_worker.cancel()
        self.colab_monitor_timer.stop()
        self._stop_global_hotkey()
        if self.local_command_server is not None:
            self.local_command_server.stop()
            self.local_command_server = None
        if self.clipboard_audio_window is not None:
            self.clipboard_audio_window.close()
        if self.colab_monitor_pool.activeThreadCount():
            self.colab_monitor_pool.waitForDone(5000)
        if self.thread_pool.activeThreadCount():
            self.statusBar().showMessage("Aguardando a tarefa atual terminar…")
            self.thread_pool.waitForDone(-1)
        self._remove_clipboard_temp_dir(self.clipboard_temp_dir)
        self.clipboard_temp_dir = None
        self.player.stop()
        self.asr.close()
        self.tts.close()
        super().closeEvent(event)


def main() -> None:
    clipboard_requested = "--clipboard-audio" in sys.argv
    if clipboard_requested:
        sys.argv = [arg for arg in sys.argv if arg != "--clipboard-audio"]
    command = "clipboard-audio" if clipboard_requested else "activate"
    if send_local_command(command):
        return

    app = QApplication(sys.argv)
    app.setApplicationName("Joe Audio Studio")
    app.setOrganizationName("g4b3r-desktop")
    icon_path = Path(__file__).resolve().parent / "resources" / "joe.png"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = JoeWindow()
    server = LocalCommandServer(window.hotkey_bridge.command_received.emit)
    try:
        server.start()
    except FileExistsError:
        if send_local_command(command):
            window.close()
            return
        window.statusBar().showMessage("Não foi possível conectar à instância aberta do Joe.", 12000)
    except (OSError, RuntimeError) as exc:
        window.statusBar().showMessage(f"Atalho IPC indisponível: {exc}", 12000)
    else:
        window.local_command_server = server

    if clipboard_requested:
        # The hotkey launcher must show only the compact audio dialog, never the main window.
        window.clipboard_only_mode = True
        QTimer.singleShot(250, window._generate_audio_from_clipboard)
    else:
        window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
