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
import zipfile
from collections.abc import Callable
from pathlib import Path

# Import PyTorch before Qt Multimedia; importing it later from a Qt worker can abort natively.
import torch  # noqa: F401

# isort: split
from PySide6.QtCore import (
    QObject, QRunnable, QRectF, QSize, QSettings, QStandardPaths, QThreadPool,
    QTime, QTimer, QUrl, Qt, Signal,
)
from PySide6.QtGui import (
    QAction, QDesktopServices, QFont, QIcon, QPalette, QPainter, QTextDocument, QTextOption,
)
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QDoubleSpinBox,
    QHeaderView,
    QInputDialog,
    QFormLayout,
    QFontComboBox,
    QFrame,
    QGridLayout,
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
    QSpinBox,
    QStatusBar,
    QStyle,
    QStyleOptionViewItem,
    QStyledItemDelegate,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from joe.audio_chunks import merge_wav_files, split_text_into_chunks, split_text_into_phrases
from joe.audio_history import AudioHistoryStore
from joe.project_archive import JOE_AUDIO_PROJECT_EXTENSION, load_joe_audio_project
from joe.generation_queue import GenerationQueue, GenerationTask
from joe.system_monitor import read_hardware_metrics
from joe.tts_profiles import build_generation_settings, performance_profile_settings
from joe.ui.audio_history import AudioHistoryPanel
from joe.ui.clipboard_audio_dialog import (
    ClipboardAudioDialog,
    DEFAULT_CLIPBOARD_DIALOG_APPEARANCE,
)
from joe.ui.advanced_player import AdvancedAudioPlayer
from joe.ui.error_dialog import ErrorDetailsDialog
from joe.audio_performance import (
    AudioPerformance,
    format_duration,
    format_elapsed_status,
    format_performance_report,
    format_tts_benchmark_report,
    format_cuda_quantization_benchmark_report,
    format_cuda_precision_benchmark_report,
    format_cuda_tf32_benchmark_report,
    format_cuda_profile_benchmark_report,
    format_cuda_attention_benchmark_report,
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
from joe.services.colab_api import ColabAPIClient, ColabAPIError, ColabVoiceSession
from joe.model_discovery import LocalModel, default_model_search_roots, discover_local_models
from joe.services.parakeet import ParakeetReduxService
from joe.services.pocket_tts import PocketTTSService
from joe.pocket_tts_models import (
    default_model_variant,
    is_model_variant_for_language,
    model_variants_for_language,
)
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
from joe.temporary_voice_cache import (
    TemporaryVoiceFileCache,
    resolve_shortcut_voice_paths,
)
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
    chunk_ready = Signal(int, str, int, str)
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
        prepare: Callable[[], object] | None = None,
    ):
        super().__init__()
        import threading

        self.chunks = list(chunks)
        self.generator = generator
        self.output_directory = output_directory
        self.final_path = final_path
        self.remote = remote
        self.prepare = prepare
        self.cancel_event = threading.Event()
        self.signals = ChunkWorkerSignals()

    def cancel(self) -> None:
        self.cancel_event.set()

    def run(self) -> None:
        started_at = time.perf_counter()
        generated: list[Path] = []
        total = len(self.chunks)

        try:
            if self.prepare is not None:
                if self.cancel_event.is_set():
                    self.signals.cancelled.emit()
                    return
                self.signals.progress.emit(
                    "Preparando áudio de referência e verificando a voz no Google Drive…"
                )
                self.prepare()

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
                self.signals.chunk_ready.emit(index, str(result), total, text)

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
            self.signals.failed.emit(
                f"{exc}\n\nTraceback técnico:\n{traceback.format_exc()}"
            )

    def _generate_chunk(
        self,
        index: int,
        total: int,
        text: str,
        destination: Path,
    ) -> Path:
        """Tenta cada bloco duas vezes antes de interromper a tarefa."""
        last_error = None
        attempt_diagnostics: list[str] = []

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
                attempt_diagnostics.append(
                    f"Tentativa {attempt + 1}: {type(exc).__name__}: {exc}\n"
                    f"{traceback.format_exc()}"
                )
                if attempt == 0:
                    self.signals.progress.emit(
                        f"Trecho {index}: falha temporária; tentando novamente…"
                    )

        raise RuntimeError(
            f"Falha no trecho {index} de {total} após duas tentativas: {last_error}"
            "\n\nDiagnóstico das tentativas:\n\n"
            + "\n\n".join(attempt_diagnostics)
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
        attempt_diagnostics: list[str] = []
        current_stage = "Preparando o texto"
        backend = "Google Colab" if self.task.remote else "Pocket TTS local"
        try:
            chunks = split_text_into_chunks(self.task.speech_text, max_chars=280)
            if not chunks:
                raise ValueError("O texto preparado para fala está vazio.")

            remote = (
                ColabAPIClient(self.task.remote_url, self.task.remote_token)
                if self.task.remote
                else None
            )
            remote_voice_id = None
            if remote is not None and self.task.voice_path:
                self.signals.progress.emit(
                    self.task.task_id,
                    "Enviando áudio de referência e preparando a voz no Google Drive…",
                )
                prepared_voice = remote.prepare_voice(
                    self.task.voice_path,
                    self.task.language,
                    self.task.model_variant,
                    name=self.task.voice_alias,
                )
                remote_voice_id = str(prepared_voice.get("voice_id") or "")
                if not remote_voice_id:
                    raise ColabAPIError(
                        "O Colab não retornou o identificador da voz preparada no Drive."
                    )
            with tempfile.TemporaryDirectory(prefix="joe-tts-queue-") as temp_dir:
                temporary_root = Path(temp_dir)
                generated: list[Path] = []

                for index, chunk_text in enumerate(chunks, start=1):
                    if self.cancel_event.is_set():
                        self.signals.cancelled.emit(self.task.task_id)
                        return

                    current_stage = f"Trecho {index} de {len(chunks)}"
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
                                    model_variant=self.task.model_variant,
                                    voice_id=remote_voice_id,
                                    sampler_decode_steps=self.task.sampler_decode_steps,
                                    temperature=self.task.temperature,
                                    eos_threshold=self.task.eos_threshold,
                                    frames_after_eos=self.task.frames_after_eos,
                                    max_tokens=self.task.max_tokens,
                                    teacher_force_previous_chunk=(
                                        self.task.teacher_force_previous_chunk
                                    ),
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
                                    self.task.model_variant,
                                    device_preference=self.task.tts_device,
                                    cpu_threads=self.task.cpu_threads,
                                    quantize=self.task.tts_quantization,
                                    sampler_decode_steps=self.task.sampler_decode_steps,
                                    temperature=self.task.temperature,
                                    eos_threshold=self.task.eos_threshold,
                                    frames_after_eos=self.task.frames_after_eos,
                                    max_tokens=self.task.max_tokens,
                                    teacher_force_previous_chunk=(
                                        self.task.teacher_force_previous_chunk
                                    ),
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
                            attempt_diagnostics.append(
                                f"{current_stage}, tentativa {attempt + 1}: "
                                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                            )
                        except Exception as exc:  # noqa: BLE001
                            if self.cancel_event.is_set():
                                self.signals.cancelled.emit(self.task.task_id)
                                return
                            last_error = exc
                            attempt_diagnostics.append(
                                f"{current_stage}, tentativa {attempt + 1}: "
                                f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
                            )

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
                technical_trace = traceback.format_exc()
                voice_label = self.task.voice_alias or (
                    "Voz personalizada" if self.task.voice_path else "Voz padrão"
                )
                report = (
                    f"Falha ao gerar o arquivo {self.task.output_name}.\n"
                    f"Serviço: {backend}\n"
                    f"Idioma: {self.task.language}\n"
                    f"Modelo: {self.task.model_variant or self.task.language}\n"
                    f"Voz: {voice_label}\n"
                    f"Etapa: {current_stage}\n"
                    f"Erro: {type(exc).__name__}: {exc}\n\n"
                    f"Traceback técnico:\n{technical_trace}"
                )
                if attempt_diagnostics:
                    report += (
                        "\n\nDetalhes das tentativas anteriores e atuais:\n\n"
                        + "\n\n".join(attempt_diagnostics)
                    )
                self.signals.failed.emit(self.task.task_id, report)


class ColabModelSelectionDialog(QDialog):
    """Discover local Pocket TTS models and let the user select Drive uploads."""

    def __init__(self, search_roots: list[Path], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Modelos Pocket TTS encontrados")
        self.setWindowIcon(QApplication.windowIcon())
        self.resize(820, 610)
        self.search_roots = list(dict.fromkeys(Path(path).expanduser() for path in search_roots))
        self.models: list[LocalModel] = []

        layout = QVBoxLayout(self)
        heading = QLabel("Escolha os modelos que deseja enviar ao Google Drive")
        heading.setObjectName("pageTitle")
        layout.addWidget(heading)
        description = QLabel(
            "O Joe procura automaticamente arquivos model.safetensors nas pastas abaixo, "
            "identifica as variantes conhecidas e prepara o envio em blocos. "
            "Modelos já existentes no cache do Hugging Face podem obter o tokenizer "
            "da mesma revisão automaticamente."
        )
        description.setWordWrap(True)
        description.setObjectName("muted")
        layout.addWidget(description)

        roots_header = QHBoxLayout()
        roots_header.addWidget(QLabel("Pastas pesquisadas"))
        roots_header.addStretch(1)
        self.add_folder_button = QPushButton("Adicionar pasta…")
        self.remove_folder_button = QPushButton("Remover selecionada")
        self.refresh_button = QPushButton("Atualizar busca")
        roots_header.addWidget(self.add_folder_button)
        roots_header.addWidget(self.remove_folder_button)
        roots_header.addWidget(self.refresh_button)
        layout.addLayout(roots_header)

        self.roots_list = QListWidget()
        self.roots_list.setMaximumHeight(105)
        for root in self.search_roots:
            self.roots_list.addItem(str(root))
        layout.addWidget(self.roots_list)

        self.scan_status = QLabel("Procurando modelos locais…")
        self.scan_status.setObjectName("muted")
        layout.addWidget(self.scan_status)

        self.model_list = QListWidget()
        self.model_list.setWordWrap(True)
        self.model_list.setSpacing(4)
        self.model_list.setUniformItemSizes(False)
        layout.addWidget(self.model_list, 1)

        footer = QHBoxLayout()
        self.select_all_button = QPushButton("Marcar todos disponíveis")
        self.clear_button = QPushButton("Desmarcar todos")
        self.upload_button = QPushButton("Enviar selecionados (0)")
        self.upload_button.setObjectName("primaryButton")
        self.upload_button.setEnabled(False)
        self.close_button = QPushButton("Fechar")
        footer.addWidget(self.select_all_button)
        footer.addWidget(self.clear_button)
        footer.addStretch(1)
        footer.addWidget(self.upload_button)
        footer.addWidget(self.close_button)
        layout.addLayout(footer)

        self.add_folder_button.clicked.connect(self._add_folder)
        self.remove_folder_button.clicked.connect(self._remove_folder)
        self.refresh_button.clicked.connect(self.refresh_models)
        self.select_all_button.clicked.connect(self._select_all)
        self.clear_button.clicked.connect(self._clear_all)
        self.model_list.itemChanged.connect(self._update_upload_button)
        self.upload_button.clicked.connect(self._accept_if_selected)
        self.close_button.clicked.connect(self.reject)

        self.refresh_models()

    @staticmethod
    def _language_name(language: str) -> str:
        return {
            "portuguese": "Português",
            "english": "Inglês",
            "spanish": "Espanhol",
            "french": "Francês",
            "italian": "Italiano",
            "german": "Alemão",
            "dutch": "Neerlandês",
        }.get(language, language.capitalize())

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        value = float(size_bytes)
        for unit in ("B", "KB", "MB", "GB"):
            if value < 1024 or unit == "GB":
                return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
            value /= 1024
        return f"{value:.1f} GB"

    def _add_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self, "Adicionar pasta para procurar modelos", str(Path.home())
        )
        if not selected:
            return
        path = Path(selected).expanduser()
        if path not in self.search_roots:
            self.search_roots.append(path)
            self.roots_list.addItem(str(path))
            self.refresh_models()

    def _remove_folder(self) -> None:
        row = self.roots_list.currentRow()
        if row < 0:
            return
        self.search_roots.pop(row)
        self.roots_list.takeItem(row)
        self.refresh_models()

    def refresh_models(self) -> None:
        self.model_list.clear()
        self.models = discover_local_models(self.search_roots)
        if not self.models:
            self.scan_status.setText(
                "Nenhum modelo Pocket TTS compatível foi encontrado. "
                "Adicione a pasta do cache/modelo ou atualize a busca."
            )
            self.model_list.addItem(
                "Nenhum modelo encontrado. O Joe reconhece variantes cadastradas "
                "que contenham model.safetensors."
            )
            self.model_list.item(0).setFlags(Qt.ItemFlag.NoItemFlags)
            self._update_upload_button()
            return

        ready_count = sum(model.can_upload for model in self.models)
        self.scan_status.setText(
            f"{len(self.models)} modelo(s) identificado(s); "
            f"{ready_count} pronto(s) para envio."
        )
        for model in self.models:
            tokenizer_state = (
                "tokenizer local disponível"
                if model.tokenizer_local
                else (
                    "tokenizer será obtido da mesma revisão Hugging Face"
                    if model.tokenizer_auto_download
                    else "tokenizer.json ausente — envio indisponível"
                )
            )
            item = QListWidgetItem(
                f"{self._language_name(model.language)} · {model.display_name} "
                f"({model.model_variant})\n"
                f"{self._format_size(model.size_bytes)} · {tokenizer_state}\n"
                f"{model.checkpoint}"
            )
            item.setData(Qt.ItemDataRole.UserRole, model)
            item.setSizeHint(QSize(0, 82))
            item.setToolTip(str(model.checkpoint))
            flags = item.flags()
            if model.can_upload:
                item.setFlags(flags | Qt.ItemFlag.ItemIsUserCheckable)
                item.setCheckState(Qt.CheckState.Unchecked)
            else:
                item.setFlags(flags & ~Qt.ItemFlag.ItemIsEnabled)
            self.model_list.addItem(item)
        self._update_upload_button()

    def _selected_models(self) -> list[LocalModel]:
        selected: list[LocalModel] = []
        for index in range(self.model_list.count()):
            item = self.model_list.item(index)
            if item.checkState() == Qt.CheckState.Checked:
                model = item.data(Qt.ItemDataRole.UserRole)
                if isinstance(model, LocalModel):
                    selected.append(model)
        return selected

    def selected_models(self) -> list[LocalModel]:
        return self._selected_models()

    def _update_upload_button(self, *_args) -> None:
        count = len(self._selected_models())
        self.upload_button.setText(f"Enviar selecionados ({count})")
        self.upload_button.setEnabled(count > 0)

    def _select_all(self) -> None:
        for index in range(self.model_list.count()):
            item = self.model_list.item(index)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Checked)

    def _clear_all(self) -> None:
        for index in range(self.model_list.count()):
            item = self.model_list.item(index)
            if item.flags() & Qt.ItemFlag.ItemIsUserCheckable:
                item.setCheckState(Qt.CheckState.Unchecked)

    def _accept_if_selected(self) -> None:
        if not self._selected_models():
            QMessageBox.information(
                self, "Selecione modelos", "Marque ao menos um modelo para enviar."
            )
            return
        self.accept()


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
        layout.addWidget(self.task_list, 3)

        self.details_label = QLabel("Detalhes da tarefa selecionada")
        layout.addWidget(self.details_label)
        self.error_details = QTextEdit()
        self.error_details.setReadOnly(True)
        self.error_details.setPlaceholderText(
            "Selecione uma tarefa para consultar o status ou os detalhes do erro."
        )
        self.error_details.setMinimumHeight(150)
        self.error_details.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.error_details.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.error_details.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        layout.addWidget(self.error_details, 1)

        detail_actions = QHBoxLayout()
        self.copy_error_button = QPushButton("Copiar erro completo")
        self.open_error_button = QPushButton("Abrir erro em janela")
        detail_actions.addWidget(self.copy_error_button)
        detail_actions.addWidget(self.open_error_button)
        detail_actions.addStretch(1)
        layout.addLayout(detail_actions)
        self.copy_error_button.clicked.connect(self._copy_selected_error)
        self.open_error_button.clicked.connect(self._open_selected_error)
        self.task_list.currentItemChanged.connect(self._refresh_task_details)

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

    def _refresh_task_details(self, *_args) -> None:
        task = self.queue.get(self._selected_task_id() or "")
        if task is None:
            self.details_label.setText("Detalhes da tarefa selecionada")
            self.error_details.clear()
            return
        if task.error:
            self.details_label.setText("Erro de geração — detalhes técnicos")
            self.error_details.setPlainText(task.error)
            self.copy_error_button.setEnabled(True)
            self.open_error_button.setEnabled(True)
        else:
            self.details_label.setText("Status da tarefa")
            self.error_details.setPlainText(task.progress or "Sem detalhes disponíveis.")
            self.copy_error_button.setEnabled(False)
            self.open_error_button.setEnabled(False)

    def _selected_error_text(self) -> str:
        task = self.queue.get(self._selected_task_id() or "")
        return task.error if task is not None and task.error else self.error_details.toPlainText()

    def _copy_selected_error(self) -> None:
        details = self._selected_error_text()
        if details.strip():
            QApplication.clipboard().setText(details)

    def _open_selected_error(self) -> None:
        details = self._selected_error_text()
        if details.strip():
            dialog = ErrorDetailsDialog("Detalhes completos do erro", details, self)
            dialog.exec()

    def select_task(self, task_id: str) -> None:
        for index in range(self.task_list.count()):
            item = self.task_list.item(index)
            if str(item.data(Qt.ItemDataRole.UserRole)) == task_id:
                self.task_list.setCurrentItem(item)
                self.task_list.scrollToItem(item)
                self._refresh_task_details()
                return

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
        items_by_id = {}
        tasks = self.queue.tasks()
        for task in tasks:
            prefix = "⚡ Urgente · " if task.urgent else ""
            label = labels.get(task.status, task.status)
            item = QListWidgetItem(
                f"{prefix}{task.output_name} — {label} — {task.progress}"
            )
            item.setData(Qt.ItemDataRole.UserRole, task.task_id)
            if task.error:
                item.setToolTip(task.error)
            self.task_list.addItem(item)
            items_by_id[task.task_id] = item
            if task.task_id == selected_id:
                selected_item = item
        if selected_item is None and selected_id is None:
            latest_failed = next(
                (task for task in reversed(tasks) if task.status == "failed"),
                None,
            )
            if latest_failed is not None:
                selected_item = items_by_id.get(latest_failed.task_id)
        if selected_item is not None:
            self.task_list.setCurrentItem(selected_item)
        self._refresh_task_details()

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


class WrappedTextItemDelegate(QStyledItemDelegate):
    """Paint list rows with word wrapping, including long words and paths."""

    def _document(self, text: str, font, text_color: str, width: int) -> QTextDocument:
        document = QTextDocument()
        document.setDocumentMargin(0)
        document.setDefaultFont(font)
        document.setPlainText(text)
        text_option = QTextOption()
        text_option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        document.setDefaultTextOption(text_option)
        document.setDefaultStyleSheet(f"body {{ color: {text_color}; }}")
        document.setTextWidth(max(120, width))
        return document

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        view = self.parent()
        width = option.rect.width()
        if width <= 0 and isinstance(view, QListWidget):
            width = view.viewport().width()
        if width <= 0:
            width = 520
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        document = self._document(text, option.font, "#edf2f8", width - 48)
        height = max(48, int(document.size().height()) + 28)
        return QSize(width, height)

    def paint(self, painter: QPainter, option, index) -> None:
        view_option = QStyleOptionViewItem(option)
        self.initStyleOption(view_option, index)
        text = view_option.text
        view_option.text = ""

        widget = view_option.widget
        style = widget.style() if widget is not None else QApplication.style()
        style.drawControl(
            QStyle.ControlElement.CE_ItemViewItem,
            view_option,
            painter,
            widget,
        )
        text_rect = style.subElementRect(
            QStyle.SubElement.SE_ItemViewItemText,
            view_option,
            widget,
        )
        color = view_option.palette.color(QPalette.ColorRole.Text).name()
        document = self._document(
            text,
            view_option.font,
            color,
            text_rect.width(),
        )
        painter.save()
        painter.translate(text_rect.topLeft())
        document.drawContents(
            painter,
            QRectF(0, 0, text_rect.width(), text_rect.height()),
        )
        painter.restore()


class WrappedListWidget(QListWidget):
    """A list that recalculates wrapped row heights when its viewport changes."""

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        self.doItemsLayout()



class JoeWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Joe — Audio Studio")
        self.setWindowIcon(QApplication.windowIcon())
        self.resize(980, 740)
        self.setMinimumSize(760, 600)

        self.tts = PocketTTSService()
        self.asr = ParakeetReduxService()
        self._applying_tts_profile = False
        self.app_settings = QSettings("g4b3r-desktop", "JoeAudioStudio")
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self.active_worker: FunctionWorker | None = None
        self.generation_queue = GenerationQueue()
        self.active_generation_worker: TTSGenerationWorker | None = None
        self.generation_queue_dialog: GenerationQueueDialog | None = None
        self.clipboard_generation_worker: ClipboardChunkWorker | None = None
        self.clipboard_recognition_worker: FunctionWorker | None = None
        self.clipboard_audio_window: ClipboardAudioDialog | None = None
        self.clipboard_temp_dir: Path | None = None
        self.clipboard_voice_cache = TemporaryVoiceFileCache()
        self._replacing_clipboard_dialog = False
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

        self.hardware_monitor_pool = QThreadPool(self)
        self.hardware_monitor_pool.setMaxThreadCount(1)
        self.hardware_monitor_worker: FunctionWorker | None = None
        self.hardware_monitor_timer = QTimer(self)
        self.hardware_monitor_timer.setInterval(2500)
        self.hardware_monitor_timer.timeout.connect(self._refresh_hardware_metrics)

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
        self.hardware_monitor_timer.start()
        self._refresh_hardware_metrics()

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_tts_tab(), "Texto → Áudio")
        tabs.addTab(self._build_asr_tab(), "Áudio → Texto")
        tabs.addTab(self._build_colab_tab(), "Google Colab / API")
        # Add after Colab so the persisted tab index 2 keeps its previous meaning.
        tabs.addTab(self._build_history_tab(), "Histórico")
        # The player and shortcut preferences are appended to preserve saved tab indices.
        tabs.addTab(self._build_player_tab(), "Player avançado")
        tabs.addTab(
            self._build_clipboard_dialog_settings_tab(),
            "Janela de atalho",
        )
        tabs.setCurrentIndex(self.app_settings.value("last_tab", 0, type=int))
        tabs.currentChanged.connect(self._on_main_tab_changed)
        self.setCentralWidget(tabs)
        if tabs.currentIndex() == 3:
            self._refresh_audio_history()

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
        open_project_action = QAction("Abrir projeto Joe…", self)
        open_project_action.triggered.connect(lambda: self._open_joe_audio_project())
        file_menu.addAction(open_project_action)
        file_menu.addSeparator()
        restart_hotkey_action = QAction("Reativar atalho global", self)
        restart_hotkey_action.triggered.connect(self._restart_global_hotkey)
        file_menu.addAction(restart_hotkey_action)

        help_menu = self.menuBar().addMenu("Ajuda")
        about_action = QAction("Sobre o Joe", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_clipboard_dialog_settings_tab(self) -> QWidget:
        """Visual preferences for the separate keyboard-shortcut audio window."""
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Personalize a janela de atalho")
        title.setObjectName("pageTitle")
        description = QLabel(
            "Altere a tipografia, o espaçamento e a cor de destaque da janela "
            "que aparece ao usar Ctrl+Shift+Espaço. As opções são salvas automaticamente "
            "e também atualizam uma janela de atalho que já esteja aberta."
        )
        description.setObjectName("muted")
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)

        appearance = DEFAULT_CLIPBOARD_DIALOG_APPEARANCE
        self.clipboard_font_family = QFontComboBox()
        saved_family = str(self.app_settings.value(
            "clipboard_dialog/font_family", appearance["font_family"]
        ))
        self.clipboard_font_family.setCurrentFont(QFont(saved_family))
        self.clipboard_font_family.setToolTip(
            "Escolha a família tipográfica usada nas frases e nos textos da janela."
        )

        self.clipboard_phrase_font_size = QSpinBox()
        self.clipboard_phrase_font_size.setRange(11, 22)
        self.clipboard_phrase_font_size.setSuffix(" px")
        self.clipboard_phrase_font_size.setValue(self.app_settings.value(
            "clipboard_dialog/phrase_font_size",
            int(appearance["phrase_font_size"]),
            type=int,
        ))

        self.clipboard_aux_font_size = QSpinBox()
        self.clipboard_aux_font_size.setRange(10, 18)
        self.clipboard_aux_font_size.setSuffix(" px")
        self.clipboard_aux_font_size.setValue(self.app_settings.value(
            "clipboard_dialog/auxiliary_font_size",
            int(appearance["auxiliary_font_size"]),
            type=int,
        ))

        self.clipboard_title_font_size = QSpinBox()
        self.clipboard_title_font_size.setRange(13, 24)
        self.clipboard_title_font_size.setSuffix(" px")
        self.clipboard_title_font_size.setValue(self.app_settings.value(
            "clipboard_dialog/title_font_size",
            int(appearance["title_font_size"]),
            type=int,
        ))

        self.clipboard_row_spacing = QSpinBox()
        self.clipboard_row_spacing.setRange(0, 12)
        self.clipboard_row_spacing.setSuffix(" px")
        self.clipboard_row_spacing.setValue(self.app_settings.value(
            "clipboard_dialog/row_spacing",
            int(appearance["row_spacing"]),
            type=int,
        ))
        self.clipboard_row_spacing.setToolTip(
            "Aumenta ou reduz o espaço vertical entre as linhas das frases."
        )

        self.clipboard_window_padding = QSpinBox()
        self.clipboard_window_padding.setRange(10, 30)
        self.clipboard_window_padding.setSuffix(" px")
        self.clipboard_window_padding.setValue(self.app_settings.value(
            "clipboard_dialog/window_padding",
            int(appearance["window_padding"]),
            type=int,
        ))

        self.clipboard_accent = QComboBox()
        self.clipboard_accent.addItem("Azul", "blue")
        self.clipboard_accent.addItem("Violeta", "violet")
        self.clipboard_accent.addItem("Verde-água", "teal")
        saved_accent = str(self.app_settings.value(
            "clipboard_dialog/accent", appearance["accent"]
        ))
        accent_index = self.clipboard_accent.findData(saved_accent)
        self.clipboard_accent.setCurrentIndex(max(0, accent_index))

        appearance_group = QGroupBox("Tipografia e aparência")
        appearance_form = QFormLayout(appearance_group)
        appearance_form.addRow("Família da fonte", self.clipboard_font_family)
        appearance_form.addRow("Texto das frases", self.clipboard_phrase_font_size)
        appearance_form.addRow("Textos auxiliares", self.clipboard_aux_font_size)
        appearance_form.addRow("Título", self.clipboard_title_font_size)
        appearance_form.addRow("Espaçamento entre frases", self.clipboard_row_spacing)
        appearance_form.addRow("Margem interna da janela", self.clipboard_window_padding)
        appearance_form.addRow("Cor de destaque", self.clipboard_accent)
        layout.addWidget(appearance_group)

        preview_group = QGroupBox("Prévia")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setSpacing(8)
        self.clipboard_appearance_preview_title = QLabel("JOE  ·  Áudio do atalho")
        self.clipboard_appearance_preview_phrase = QLabel(
            "001    ✓    Esta é uma frase de exemplo da janela de atalho."
        )
        self.clipboard_appearance_preview_phrase.setWordWrap(True)
        self.clipboard_appearance_preview_aux = QLabel(
            "43,4 s de áudio · gerado em 3:05 · 17 frases"
        )
        self.clipboard_appearance_preview_aux.setObjectName("muted")
        preview_layout.addWidget(self.clipboard_appearance_preview_title)
        preview_layout.addWidget(self.clipboard_appearance_preview_phrase)
        preview_layout.addWidget(self.clipboard_appearance_preview_aux)
        layout.addWidget(preview_group)

        actions = QHBoxLayout()
        self.clipboard_appearance_status = QLabel(
            "As alterações são aplicadas automaticamente."
        )
        self.clipboard_appearance_status.setObjectName("muted")
        reset_button = QPushButton("Restaurar padrão")
        reset_button.clicked.connect(self._reset_clipboard_dialog_appearance)
        actions.addWidget(self.clipboard_appearance_status, 1)
        actions.addWidget(reset_button)
        layout.addLayout(actions)
        layout.addStretch(1)

        self.clipboard_font_family.currentFontChanged.connect(
            self._save_clipboard_dialog_appearance
        )
        for control in (
            self.clipboard_phrase_font_size,
            self.clipboard_aux_font_size,
            self.clipboard_title_font_size,
            self.clipboard_row_spacing,
            self.clipboard_window_padding,
        ):
            control.valueChanged.connect(self._save_clipboard_dialog_appearance)
        self.clipboard_accent.currentIndexChanged.connect(
            self._save_clipboard_dialog_appearance
        )
        self._refresh_clipboard_appearance_preview()
        return page

    def _refresh_clipboard_appearance_preview(self) -> None:
        if not hasattr(self, "clipboard_appearance_preview_phrase"):
            return
        family = self.clipboard_font_family.currentFont().family()
        phrase_font = QFont(family)
        phrase_font.setPixelSize(self.clipboard_phrase_font_size.value())
        auxiliary_font = QFont(family)
        auxiliary_font.setPixelSize(self.clipboard_aux_font_size.value())
        title_font = QFont(family)
        title_font.setPixelSize(self.clipboard_title_font_size.value())
        title_font.setBold(True)

        self.clipboard_appearance_preview_title.setFont(title_font)
        self.clipboard_appearance_preview_phrase.setFont(phrase_font)
        self.clipboard_appearance_preview_aux.setFont(auxiliary_font)

        accent_colors = {
            "blue": "#368de0",
            "violet": "#9a83e8",
            "teal": "#36b9ad",
        }
        accent = accent_colors.get(
            str(self.clipboard_accent.currentData()), accent_colors["blue"]
        )
        self.clipboard_appearance_preview_title.setStyleSheet(
            f"background: transparent; color: {accent};"
        )
        self.clipboard_appearance_preview_phrase.setStyleSheet(
            "background: transparent; color: #edf3fb;"
        )
        self.clipboard_appearance_preview_aux.setStyleSheet(
            "background: transparent; color: #c0cede;"
        )

    def _save_clipboard_dialog_appearance(self, *_args) -> None:
        if not hasattr(self, "clipboard_font_family"):
            return
        self.app_settings.setValue(
            "clipboard_dialog/font_family",
            self.clipboard_font_family.currentFont().family(),
        )
        self.app_settings.setValue(
            "clipboard_dialog/phrase_font_size",
            self.clipboard_phrase_font_size.value(),
        )
        self.app_settings.setValue(
            "clipboard_dialog/auxiliary_font_size",
            self.clipboard_aux_font_size.value(),
        )
        self.app_settings.setValue(
            "clipboard_dialog/title_font_size",
            self.clipboard_title_font_size.value(),
        )
        self.app_settings.setValue(
            "clipboard_dialog/row_spacing",
            self.clipboard_row_spacing.value(),
        )
        self.app_settings.setValue(
            "clipboard_dialog/window_padding",
            self.clipboard_window_padding.value(),
        )
        self.app_settings.setValue(
            "clipboard_dialog/accent",
            self.clipboard_accent.currentData(),
        )
        self.app_settings.sync()
        self._refresh_clipboard_appearance_preview()

        dialog = self.clipboard_audio_window
        if dialog is not None:
            try:
                dialog.apply_appearance_settings()
            except RuntimeError:
                self.clipboard_audio_window = None
        self.clipboard_appearance_status.setText(
            "Preferências salvas e aplicadas."
        )

    def _reset_clipboard_dialog_appearance(self) -> None:
        defaults = DEFAULT_CLIPBOARD_DIALOG_APPEARANCE
        controls = (
            self.clipboard_font_family,
            self.clipboard_phrase_font_size,
            self.clipboard_aux_font_size,
            self.clipboard_title_font_size,
            self.clipboard_row_spacing,
            self.clipboard_window_padding,
            self.clipboard_accent,
        )
        for control in controls:
            control.blockSignals(True)
        self.clipboard_font_family.setCurrentFont(
            QFont(str(defaults["font_family"]))
        )
        self.clipboard_phrase_font_size.setValue(
            int(defaults["phrase_font_size"])
        )
        self.clipboard_aux_font_size.setValue(
            int(defaults["auxiliary_font_size"])
        )
        self.clipboard_title_font_size.setValue(
            int(defaults["title_font_size"])
        )
        self.clipboard_row_spacing.setValue(int(defaults["row_spacing"]))
        self.clipboard_window_padding.setValue(int(defaults["window_padding"]))
        accent_index = self.clipboard_accent.findData(defaults["accent"])
        self.clipboard_accent.setCurrentIndex(max(0, accent_index))
        for control in controls:
            control.blockSignals(False)
        self._save_clipboard_dialog_appearance()

    def _build_colab_tab(self) -> QWidget:
        page = QWidget()
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        page_layout.setSpacing(0)

        scroll_area = QScrollArea(page)
        scroll_area.setObjectName("colabScrollArea")
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )

        content = QWidget()
        content.setObjectName("colabScrollContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(24, 22, 24, 28)
        layout.setSpacing(20)
        scroll_area.setWidget(content)
        page_layout.addWidget(scroll_area)

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
        form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )
        form.setLabelAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
        )
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(14)
        form.addRow("Modo de execução", self.processing_mode)
        form.addRow("URL da API", self.colab_url)
        form.addRow("Token de acesso", self.colab_token)

        self.colab_tts_cpu_threads = QSpinBox()
        self.colab_tts_cpu_threads.setRange(1, 64)
        self.colab_tts_cpu_threads.setValue(
            self.app_settings.value("colab_tts_cpu_threads", 2, type=int)
        )
        self.colab_tts_cpu_threads.setFixedWidth(132)
        self.colab_tts_cpu_threads.setMinimumHeight(36)
        self.colab_tts_cpu_threads.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.colab_tts_cpu_threads.setButtonSymbols(
            QSpinBox.ButtonSymbols.UpDownArrows
        )
        self.colab_tts_cpu_threads.setToolTip(
            "Número de threads CPU usadas pelo Pocket TTS no runtime do Google Colab."
        )
        self.colab_tts_sampler_steps = QSpinBox()
        self.colab_tts_sampler_steps.setRange(1, 16)
        self.colab_tts_sampler_steps.setValue(
            self.app_settings.value("colab_tts_sampler_steps", 4, type=int)
        )
        self.colab_tts_sampler_steps.setFixedWidth(132)
        self.colab_tts_sampler_steps.setMinimumHeight(36)
        self.colab_tts_sampler_steps.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.colab_tts_sampler_steps.setButtonSymbols(
            QSpinBox.ButtonSymbols.UpDownArrows
        )
        self.colab_tts_sampler_steps.setToolTip(
            "Mais passos podem melhorar a qualidade da fala, mas aumentam o tempo de síntese."
        )
        self.colab_tts_cpu_threads.valueChanged.connect(self._save_colab_tts_settings)
        self.colab_tts_sampler_steps.valueChanged.connect(self._save_colab_tts_settings)

        self.colab_tts_quantize_int8 = QCheckBox(
            "Ativar quantização INT8 no Pocket TTS do Colab"
        )
        self.colab_tts_quantize_int8.setChecked(
            self.app_settings.value("colab_tts_quantize_int8", False, type=bool)
        )
        self.colab_tts_quantize_int8.setToolTip(
            "Desmarcada por padrão. Quando marcada, ativa INT8 nas camadas de atenção e FFN. "
            "Na GPU exige TorchAO; desmarcada, a inferência atual sem quantização é preservada."
        )
        self.colab_tts_quantize_int8.toggled.connect(self._save_colab_tts_settings)

        colab_tts_form = QFormLayout()
        colab_tts_form.setFieldGrowthPolicy(
            QFormLayout.FieldGrowthPolicy.FieldsStayAtSizeHint
        )
        colab_tts_form.setFormAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        colab_tts_form.setHorizontalSpacing(18)
        colab_tts_form.setVerticalSpacing(10)
        colab_tts_form.addRow("Threads CPU:", self.colab_tts_cpu_threads)
        colab_tts_form.addRow("Qualidade (passos):", self.colab_tts_sampler_steps)
        colab_tts_form.addRow("Quantização INT8:", self.colab_tts_quantize_int8)
        self.colab_tts_settings_status = QLabel(
            "Padrão: 2 threads CPU, qualidade 4 e quantização desativada. "
            "Aplique no runtime conectado para ativar."
        )
        self.colab_tts_settings_status.setObjectName("muted")
        self.colab_tts_settings_status.setWordWrap(True)
        self.colab_tts_apply_button = QPushButton("Aplicar configurações no Colab")
        self.colab_tts_apply_button.setObjectName("primaryButton")
        self.colab_tts_apply_button.setFixedWidth(260)
        self.colab_tts_apply_button.setMinimumHeight(36)
        self.colab_tts_apply_button.clicked.connect(self._apply_colab_tts_settings)

        self.colab_tts_group = QGroupBox("Desempenho e qualidade do Pocket TTS no Colab")
        colab_tts_layout = QVBoxLayout(self.colab_tts_group)
        colab_tts_layout.setContentsMargins(18, 26, 18, 18)
        colab_tts_layout.setSpacing(14)
        colab_tts_layout.addLayout(colab_tts_form)
        colab_tts_note = QLabel(
            "Threads CPU é limitado automaticamente à quantidade disponível no Colab. "
            "Mudanças de qualidade ou INT8 removem apenas o modelo do cache em memória "
            "e recarregam na próxima síntese; os modelos e as vozes no Google Drive não "
            "são modificados. Com INT8 desmarcada, o modelo segue sem quantização."
        )
        colab_tts_note.setObjectName("muted")
        colab_tts_note.setWordWrap(True)
        colab_tts_layout.addWidget(colab_tts_note)
        colab_tts_actions = QHBoxLayout()
        colab_tts_actions.addWidget(self.colab_tts_apply_button)
        colab_tts_actions.addStretch(1)
        colab_tts_layout.addLayout(colab_tts_actions)
        colab_tts_layout.addWidget(self.colab_tts_settings_status)

        self.colab_test_button = QPushButton("Testar conexão com o Colab")
        self.colab_test_button.setObjectName("primaryButton")
        self.colab_test_button.clicked.connect(self._test_colab_connection)

        self.colab_upload_model_button = QPushButton(
            "Localizar modelos e escolher o que enviar…"
        )
        self.colab_upload_model_button.setToolTip(
            "O Joe procura automaticamente os modelos Pocket TTS no cache Hugging Face "
            "e nas pastas adicionadas. Selecione um ou mais modelos para enviar ao Drive."
        )
        self.colab_upload_model_button.clicked.connect(self._upload_colab_model)

        self.colab_open_button = QPushButton("Abrir notebook no Google Colab")
        self.colab_open_button.clicked.connect(self._open_colab_notebook)

        self.colab_status = QLabel("Estado da API: não configurado")
        self.colab_status.setObjectName("muted")
        self.colab_status.setWordWrap(True)

        self.colab_info = QLabel(
            "1. Abra o notebook do branch de armazenamento direto e execute as células.\n"
            "2. Copie a URL e o token exibidos no final e cole nos campos acima.\n"
            "3. Ajuste Threads CPU e Qualidade nesta tela e clique em 'Aplicar configurações no Colab'.\n"
            "4. Clique em 'Localizar modelos e escolher o que enviar' para instalar modelos no Drive.\n"
            "5. Selecione um áudio original no Joe; ele será enviado e preparado no Colab.\n\n"
            "Modelos, áudios originais e estados de voz ficam em arquivos individuais no Google Drive. "
            "O novo fluxo não utiliza JOE_MODELS.zip.\n\n"
            "O Joe verifica a conexão a cada 60 segundos enquanto o modo Colab está selecionado. "
            "Essa verificação apenas detecta desconexões: não impede o Colab de encerrar o runtime "
            "por inatividade ou por seus limites de sessão.\n"
            "A URL e o token são lembrados localmente entre execuções. O token fica salvo "
            "nas configurações do usuário sem criptografia; não compartilhe o arquivo de "
            "configuração do Joe."
        )
        self.colab_info.setObjectName("muted")
        self.colab_info.setWordWrap(True)

        self.colab_runtime_group = QGroupBox("Estado e informações da API")
        colab_runtime_layout = QVBoxLayout(self.colab_runtime_group)
        colab_runtime_layout.setContentsMargins(18, 26, 18, 18)
        colab_runtime_layout.setSpacing(12)
        colab_runtime_layout.addWidget(self.colab_status)
        colab_runtime_layout.addWidget(self.colab_info)

        self.colab_url.editingFinished.connect(self._save_colab_settings)

        buttons = QHBoxLayout()
        buttons.setSpacing(12)
        buttons.addWidget(self.colab_test_button)
        buttons.addWidget(self.colab_open_button)
        buttons.addStretch(1)

        connection_group = QGroupBox("Conexão com o Google Colab")
        connection_layout = QVBoxLayout(connection_group)
        connection_layout.setContentsMargins(18, 26, 18, 18)
        connection_layout.setSpacing(16)
        connection_layout.addLayout(form)
        connection_layout.addLayout(buttons)

        model_upload_group = QGroupBox("Modelos no Google Drive — sem ZIP")
        model_upload_layout = QVBoxLayout(model_upload_group)
        model_upload_layout.setContentsMargins(18, 26, 18, 18)
        model_upload_layout.setSpacing(14)
        model_upload_description = QLabel(
            "O Joe identifica automaticamente os modelos Pocket TTS no cache local. "
            "Na janela seguinte, marque apenas as variantes que deseja enviar. "
            "É possível adicionar outras pastas e enviar vários modelos em sequência; "
            "não é necessário selecionar os arquivos individualmente."
        )
        model_upload_description.setObjectName("muted")
        model_upload_description.setWordWrap(True)
        model_upload_layout.addWidget(model_upload_description)
        model_upload_layout.addWidget(self.colab_upload_model_button)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(connection_group)
        layout.addWidget(self.colab_tts_group)
        layout.addWidget(model_upload_group)
        layout.addWidget(self.colab_runtime_group)
        layout.addStretch(1)

        self._update_mode_widgets()
        return page

    def _colab_model_search_roots(self) -> list[Path]:
        roots = default_model_search_roots()
        saved = self.app_settings.value("colab_model_search_paths", [])
        if isinstance(saved, str):
            saved = [saved] if saved.strip() else []
        if isinstance(saved, (list, tuple)):
            roots.extend(Path(value).expanduser() for value in saved if str(value).strip())
        unique: list[Path] = []
        seen: set[str] = set()
        for root in roots:
            try:
                resolved = root.resolve()
            except OSError:
                resolved = root.absolute()
            key = str(resolved)
            if key not in seen:
                seen.add(key)
                unique.append(resolved)
        return unique

    def _upload_colab_model(self) -> None:
        dialog = ColabModelSelectionDialog(self._colab_model_search_roots(), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        selected_models = dialog.selected_models()
        if not selected_models:
            return

        self.app_settings.setValue(
            "colab_model_search_paths", [str(path) for path in dialog.search_roots]
        )
        try:
            client = self._colab_client()
        except ValueError as exc:
            QMessageBox.information(self, "Configure a API do Colab", str(exc))
            return

        def upload_selected() -> dict:
            outcomes = []
            for model in selected_models:
                try:
                    result = client.upload_model_bundle(
                        model.upload_directory, model.language, model.model_variant
                    )
                    manifest = (
                        result.get("manifest")
                        if isinstance(result, dict) and isinstance(result.get("manifest"), dict)
                        else {}
                    )
                    transferred = int(manifest.get("checkpoint_size", model.size_bytes))
                    transferred += int(manifest.get("tokenizer_size", 0))
                    outcomes.append({
                        "model_variant": model.model_variant,
                        "success": True,
                        "size_bytes": transferred,
                    })
                except Exception as exc:  # noqa: BLE001
                    outcomes.append({
                        "model_variant": model.model_variant,
                        "success": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    })
            return {"outcomes": outcomes}

        self._run_task(
            f"Enviando {len(selected_models)} modelo(s) selecionado(s) ao Google Drive…",
            upload_selected,
            self._colab_models_uploaded,
        )

    def _colab_models_uploaded(self, result: object) -> None:
        data = result if isinstance(result, dict) else {}
        outcomes = data.get("outcomes", [])
        if not isinstance(outcomes, list):
            outcomes = []
        successes = [item for item in outcomes if isinstance(item, dict) and item.get("success")]
        failures = [item for item in outcomes if isinstance(item, dict) and not item.get("success")]
        lines = []
        for item in successes:
            size_mb = int(item.get("size_bytes", 0)) / (1024 * 1024)
            lines.append(f"✓ {item.get('model_variant')}: enviado ({size_mb:.1f} MB)")
        for item in failures:
            lines.append(
                f"✗ {item.get('model_variant')}: {item.get('error', 'falha sem detalhes')}"
            )
        message = "\n".join(lines) or "Nenhum modelo foi enviado."
        if failures:
            QMessageBox.warning(
                self,
                "Envio de modelos concluído com pendências",
                f"Enviados: {len(successes)} · Falhas: {len(failures)}\n\n{message}",
            )
        else:
            QMessageBox.information(
                self,
                "Modelos instalados no Google Drive",
                f"{len(successes)} modelo(s) enviado(s) e validado(s):\n\n{message}",
            )
        self.statusBar().showMessage(
            f"Envio de modelos concluído: {len(successes)} sucesso(s), "
            f"{len(failures)} falha(s).",
            9000,
        )

    def _colab_model_uploaded(self, result: object) -> None:
        data = result if isinstance(result, dict) else {}
        variant = str(data.get("model_variant") or "modelo")
        manifest = data.get("manifest") if isinstance(data.get("manifest"), dict) else {}
        size_mb = (
            int(manifest.get("checkpoint_size", 0))
            + int(manifest.get("tokenizer_size", 0))
        ) / (1024 * 1024)
        QMessageBox.information(
            self,
            "Modelo instalado no Google Drive",
            f"Variante: {variant}\n"
            f"Tamanho transferido: {size_mb:.1f} MB\n"
            "O modelo foi validado e salvo em arquivos individuais no Drive. "
            "Vozes selecionadas serão preparadas pelo próprio Colab durante a geração.",
        )
        self.statusBar().showMessage(
            f"Modelo {variant} instalado no Google Drive.", 7000
        )

    def _save_colab_tts_settings(self, *_args) -> None:
        if hasattr(self, "colab_tts_cpu_threads"):
            self.app_settings.setValue(
                "colab_tts_cpu_threads", int(self.colab_tts_cpu_threads.value())
            )
        if hasattr(self, "colab_tts_sampler_steps"):
            self.app_settings.setValue(
                "colab_tts_sampler_steps", int(self.colab_tts_sampler_steps.value())
            )
        if hasattr(self, "colab_tts_quantize_int8"):
            self.app_settings.setValue(
                "colab_tts_quantize_int8", bool(self.colab_tts_quantize_int8.isChecked())
            )
        if hasattr(self, "colab_tts_settings_status"):
            self.colab_tts_settings_status.setText(
                "Configurações salvas localmente. Clique em 'Aplicar configurações no Colab' "
                "para enviá-las ao runtime conectado."
            )

    def _apply_colab_tts_settings(self) -> None:
        if getattr(self, "colab_tts_settings_worker", None) is not None:
            self.statusBar().showMessage(
                "A aplicação das configurações do Colab já está em andamento.", 4000
            )
            return
        try:
            client = self._colab_client()
        except ValueError as exc:
            QMessageBox.information(self, "Configure a API do Colab", str(exc))
            return

        cpu_threads = int(self.colab_tts_cpu_threads.value())
        sampler_steps = int(self.colab_tts_sampler_steps.value())
        quantize_int8 = bool(self.colab_tts_quantize_int8.isChecked())
        self._save_colab_tts_settings()
        self.colab_tts_apply_button.setEnabled(False)
        self.colab_tts_settings_status.setText(
            "Enviando as configurações para o Google Colab…"
        )
        worker = FunctionWorker(
            lambda: client.update_tts_settings(
                cpu_threads, sampler_steps, quantize_int8=quantize_int8
            )
        )
        self.colab_tts_settings_worker = worker
        worker.signals.finished.connect(self._colab_tts_settings_applied)
        worker.signals.failed.connect(self._colab_tts_settings_failed)
        self.colab_monitor_pool.start(worker)

    def _colab_tts_settings_applied(self, result: object) -> None:
        self.colab_tts_settings_worker = None
        self.colab_tts_apply_button.setEnabled(True)
        data = result if isinstance(result, dict) else {}
        applied_threads = data.get("tts_cpu_threads", "—")
        requested_threads = data.get("tts_requested_cpu_threads", applied_threads)
        available_threads = data.get("tts_available_cpu_threads", "—")
        sampler_steps = data.get(
            "tts_sampler_decode_steps", self.colab_tts_sampler_steps.value()
        )
        quantization_label = (
            "INT8 solicitada" if data.get("tts_quantize_int8_requested")
            else "sem quantização"
        )
        message = (
            f"Aplicado no Colab: {applied_threads} thread(s) CPU "
            f"(de {available_threads} disponível(is)), qualidade {sampler_steps} "
            f"e {quantization_label}."
        )
        if requested_threads != applied_threads:
            message += (
                f" O Colab ajustou o valor solicitado ({requested_threads}) "
                "ao limite de CPU do runtime."
            )
        if data.get("tts_model_reloaded_on_next_generation"):
            message += " O modelo será recarregado na próxima síntese."
        self.colab_tts_settings_status.setText(message)
        self.statusBar().showMessage("Configurações do Pocket TTS aplicadas no Colab.", 6000)
        self._start_colab_check(manual=False)

    def _colab_tts_settings_failed(self, message: str) -> None:
        self.colab_tts_settings_worker = None
        self.colab_tts_apply_button.setEnabled(True)
        detail = str(message).split("\n\n", 1)[0].strip()
        self.colab_tts_settings_status.setText(
            f"Não foi possível aplicar as configurações: {detail}"
        )
        self.statusBar().showMessage("Falha ao aplicar configurações no Colab.", 6000)

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
        gpu = data.get("gpu_name") or ("CUDA disponível" if data.get("cuda_available") else "não disponível")
        tts_mode = data.get("tts_device_mode", "auto")
        tts_devices = data.get("tts_loaded_devices") or {}
        tts_device = next(iter(tts_devices.values()), data.get("tts_device_selected", "não selecionado"))
        tts_threads = data.get("tts_cpu_threads", "—")
        tts_available_threads = data.get("tts_available_cpu_threads", "—")
        tts_steps = data.get("tts_sampler_decode_steps", "—")
        tts_quant_requested = bool(data.get("tts_quantize_int8_requested", False))
        tts_quant_active = bool(data.get("tts_quantize_int8_active", False))
        tts_quant_devices = data.get("tts_quantization_status") or {}
        tts_quant_summary = ", ".join(
            f"{variant}: {details.get('backend', '—')} em {details.get('device', '—')}"
            for variant, details in tts_quant_devices.items()
            if isinstance(details, dict) and details.get("active")
        )
        last_tts = data.get("last_tts_generation") or {}
        last_tts_status = (
            f"Última síntese: {last_tts.get('device')} | "
            f"fluxo: {last_tts.get('flow_type', 'N/D')} | "
            f"sampler: {last_tts.get('sampler_decode_steps', 'N/D')} passo(s) | "
            f"RTF {float(last_tts.get('realtime_factor', 0)):.2f} | "
            f"{float(last_tts.get('duration_seconds', 0)):.2f}s de áudio."
            if last_tts else "Ainda não há síntese TTS nesta sessão."
        )
        pocket_tts_version = data.get("pocket_tts_version", "N/D")
        loaded_model_diagnostics = data.get("tts_loaded_model_diagnostics") or {}
        loaded_model_summary = ", ".join(
            f"{variant}: fluxo {details.get('flow_type', 'N/D')} / "
            f"{details.get('sampler_decode_steps', 'N/D')} passo(s)"
            for variant, details in loaded_model_diagnostics.items()
            if isinstance(details, dict)
        ) or "nenhum modelo carregado"
        checked_at = QTime.currentTime().toString("HH:mm:ss")
        self.colab_status.setText(f"API online — última verificação às {checked_at}.")
        direct_drive = bool(data.get("direct_drive_assets_enabled"))
        installed_models = data.get("tts_direct_drive_models") or []
        voice_count = int(data.get("tts_direct_drive_voice_count") or 0)
        storage_status = (
            f"Armazenamento Drive direto: ativo — modelos: {', '.join(installed_models) or 'nenhum'}; "
            f"vozes preparadas: {voice_count}."
            if direct_drive else
            "Armazenamento Drive direto: indisponível. Abra o notebook novo do branch "
            "feature/colab-drive-direct-assets e aplique a montagem do Drive."
        )
        self.colab_info.setText(
            f"Conexão confirmada. Servidor: {data.get('service', 'Joe Colab API')}\n"
            f"ASR: {data.get('asr', 'Parakeet Redux')}\n"
            f"GPU disponível: {gpu}\n"
            f"Pocket TTS: versão {pocket_tts_version} | {tts_device} (modo {tts_mode}) | "
            f"threads CPU: {tts_threads}/{tts_available_threads} | qualidade configurada: {tts_steps}\n"
            f"Quantização INT8: {'solicitada' if tts_quant_requested else 'desativada'} "
            f"({'ativa: ' + tts_quant_summary if tts_quant_active else 'ainda não aplicada nesta sessão'})\n"
            f"Modelos carregados: {loaded_model_summary}\n"
            f"{last_tts_status}\n"
            f"Parakeet carregado: {'sim' if data.get('asr_loaded') else 'ainda não'}\n"
            f"{storage_status}\n\n"
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
            "blob/feature/pocket-tts-generation-controls/colab/Joe_Colab_API.ipynb"
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
        self.tts_language.addItem("Japonês (experimental)", "japanese")
        saved_language = str(self.app_settings.value("tts_language", "portuguese"))
        language_index = self.tts_language.findData(saved_language)
        self.tts_language.setCurrentIndex(max(0, language_index))

        raw_model_preferences = self.app_settings.value("tts_model_variants", "{}")
        try:
            self._tts_model_preferences = json.loads(str(raw_model_preferences))
        except (TypeError, ValueError):
            self._tts_model_preferences = {}
        if not isinstance(self._tts_model_preferences, dict):
            self._tts_model_preferences = {}
        saved_model_variant = str(
            self.app_settings.value("tts_model_variant", default_model_variant(saved_language))
        )
        if (
            saved_language not in self._tts_model_preferences
            and is_model_variant_for_language(saved_language, saved_model_variant)
        ):
            self._tts_model_preferences[saved_language] = saved_model_variant

        self.tts_model_variant = QComboBox()
        self.tts_model_variant.setToolTip(
            "Escolha a variante do Pocket TTS usada nas próximas gerações."
        )
        self.tts_model_variant.currentIndexChanged.connect(self._save_tts_model_variant)
        self.tts_model_variant_info = QLabel()
        self.tts_model_variant_info.setObjectName("muted")
        self.tts_model_variant_info.setWordWrap(True)
        self._refresh_tts_model_variants()
        self.tts_language.currentIndexChanged.connect(self._on_tts_language_changed)

        self.voice_path = QLineEdit(str(self.app_settings.value("voice_path", "")))
        self.voice_path.setReadOnly(True)
        self.voice_path.setPlaceholderText("Voz padrão do Pocket TTS")
        self.voice_path.textChanged.connect(
            lambda value: self._save_text_setting("voice_path", value)
        )
        self.voice_path.textChanged.connect(self._sync_favorite_editor)
        self.voice_path.textChanged.connect(self._on_clipboard_voice_source_changed)

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
        voice_form.addRow("Versão do Pocket TTS", self.tts_model_variant)
        voice_form.addRow("Detalhes da versão", self.tts_model_variant_info)
        voice_form.addRow("Pasta da biblioteca", library_row)
        voice_form.addRow("Voz padrão / referência", voice_row)
        settings_layout.addWidget(voice_group)

        # Local Pocket TTS compute settings and an on-device CPU/CUDA comparison.
        hardware_group = QGroupBox("Hardware e desempenho do Pocket TTS")
        hardware_form = QFormLayout(hardware_group)
        self.tts_performance_profile = QComboBox()
        for profile_key, profile_label in (
            ("custom", "Personalizado"),
            ("economy", "Econômico"),
            ("balanced", "Equilibrado"),
            ("performance", "Desempenho"),
        ):
            self.tts_performance_profile.addItem(profile_label, profile_key)
        saved_profile = str(
            self.app_settings.value("tts_performance_profile", "custom")
        )
        saved_profile_index = self.tts_performance_profile.findData(saved_profile)
        self.tts_performance_profile.setCurrentIndex(max(0, saved_profile_index))
        self.tts_performance_profile.setToolTip(
            "Os perfis preenchem dispositivo, threads e quantização. "
            "Você pode alterar cada opção depois; isso muda o perfil para Personalizado."
        )

        self.tts_device = QComboBox()
        self.tts_device.addItem("CPU (processador)", "cpu")
        self.tts_device.addItem("NVIDIA GPU (CUDA — usa VRAM)", "cuda")
        saved_tts_device = str(self.app_settings.value("tts_device", "cpu"))
        saved_device_index = self.tts_device.findData(saved_tts_device)
        self.tts_device.setCurrentIndex(max(0, saved_device_index))
        self.tts_device.setToolTip(
            "CPU e CUDA têm controles INT8 separados. A quantização da GPU começa "
            "desativada e só é usada quando você a habilita explicitamente."
        )

        self.tts_cpu_threads = QSpinBox()
        self.tts_cpu_threads.setRange(1, max(1, min(128, os.cpu_count() or 1)))
        saved_cpu_threads = self.app_settings.value("tts_cpu_threads", 2, type=int)
        self.tts_cpu_threads.setValue(
            max(self.tts_cpu_threads.minimum(), min(self.tts_cpu_threads.maximum(), saved_cpu_threads))
        )
        self.tts_cpu_threads.setToolTip(
            "Quantidade de threads intra-op do PyTorch. A alteração é aplicada às próximas "
            "gerações locais; valores acima do número de núcleos disponíveis não são oferecidos."
        )

        self.tts_quantization = QCheckBox("Usar quantização dinâmica INT8 na CPU")
        saved_quantization = self.app_settings.value(
            "tts_quantization", True, type=bool
        )
        self.tts_quantization.setChecked(saved_quantization)
        self.tts_quantization.setToolTip(
            "Usa o caminho INT8 dinâmico padrão do Pocket TTS na CPU. "
            "Esta preferência não ativa a quantização na GPU."
        )

        self.tts_cuda_quantization = QCheckBox(
            "Usar quantização INT8 na NVIDIA GPU (TorchAO)"
        )
        saved_cuda_quantization = self.app_settings.value(
            "tts_cuda_quantization", False, type=bool
        )
        self.tts_cuda_quantization.setChecked(saved_cuda_quantization)
        self.tts_cuda_quantization.setToolTip(
            "Ativa INT8 com TorchAO nas camadas de atenção e FFN do Pocket TTS na CUDA. "
            "Desmarcada por padrão: a inferência GPU continua exatamente sem quantização."
        )

        self.tts_hardware_info = QLabel()
        self.tts_hardware_info.setObjectName("muted")
        self.tts_hardware_info.setWordWrap(True)
        self.tts_device.currentIndexChanged.connect(self._save_tts_hardware_settings)
        self.tts_cpu_threads.valueChanged.connect(self._save_tts_hardware_settings)
        self.tts_quantization.toggled.connect(self._save_tts_hardware_settings)
        self.tts_cuda_quantization.toggled.connect(self._save_tts_hardware_settings)
        self.tts_performance_profile.currentIndexChanged.connect(
            self._apply_tts_performance_profile
        )

        self.tts_benchmark_button = QPushButton("Comparar CPU × CUDA")
        self.tts_benchmark_button.setToolTip(
            "Executa o mesmo texto e voz em CPU e CUDA, medindo carga, preparação da voz, "
            "geração, velocidade em tempo real e pico de VRAM."
        )
        self.tts_benchmark_button.clicked.connect(self._benchmark_tts_performance)

        self.tts_cuda_quant_benchmark_runs = QSpinBox()
        self.tts_cuda_quant_benchmark_runs.setRange(3, 15)
        self.tts_cuda_quant_benchmark_runs.setValue(
            self.app_settings.value("tts_cuda_quant_benchmark_runs", 7, type=int)
        )
        self.tts_cuda_quant_benchmark_runs.setFixedWidth(72)
        self.tts_cuda_quant_benchmark_runs.setToolTip(
            "Quantidade de gerações medidas por modo, depois de duas gerações de aquecimento."
        )
        self.tts_cuda_quant_benchmark_runs.valueChanged.connect(
            lambda value: self.app_settings.setValue(
                "tts_cuda_quant_benchmark_runs", int(value)
            )
        )
        self.tts_cuda_quant_benchmark_button = QPushButton(
            "Medir CUDA: INT8 × sem INT8"
        )
        self.tts_cuda_quant_benchmark_button.setToolTip(
            "Aquece cada modo, descarta as primeiras duas gerações e mede várias sínteses. "
            "Mostra tempos individuais, média, mediana, desvio padrão e VRAM."
        )
        self.tts_cuda_quant_benchmark_button.clicked.connect(
            self._benchmark_tts_cuda_quantization
        )
        self.tts_cuda_fp16_benchmark_button = QPushButton(
            "Medir CUDA: FP32 × FP16"
        )
        self.tts_cuda_fp16_benchmark_button.setToolTip(
            "Compara CUDA FP32 com FP16 autocast experimental nos três pontos internos "
            "do Pocket TTS. Mantém os pesos FP32 e não altera as próximas gerações."
        )
        self.tts_cuda_fp16_benchmark_button.clicked.connect(
            self._benchmark_tts_cuda_precision
        )
        self.tts_cuda_tf32_benchmark_button = QPushButton(
            "Medir CUDA: FP32 IEEE × TF32"
        )
        self.tts_cuda_tf32_benchmark_button.setToolTip(
            "Compara FP32 IEEE com TF32 em multiplicações CUDA na GPU Ampere+. "
            "Pesos e saídas permanecem FP32; a configuração original é restaurada no final."
        )
        self.tts_cuda_tf32_benchmark_button.clicked.connect(
            self._benchmark_tts_cuda_tf32
        )
        self.tts_cuda_profile_button = QPushButton(
            "Diagnosticar CUDA: Flow LM × Mimi"
        )
        self.tts_cuda_profile_button.setToolTip(
            "Mede separadamente Flow LM, decoder Mimi, espera na fila de latentes "
            "e transferência do áudio para a RAM. Usa CUDA FP32 sem mudar as gerações normais."
        )
        self.tts_cuda_profile_button.clicked.connect(
            self._benchmark_tts_cuda_profile
        )
        self.tts_cuda_attention_benchmark_button = QPushButton(
            "Comparar SDPA FP32 × fused FP16"
        )
        self.tts_cuda_attention_benchmark_button.setToolTip(
            "Compara a atenção SDPA padrão FP32 com Flash e Memory-Efficient FP16 somente nos "
            "tensores Q/K/V. Projeções e restante do Flow LM continuam FP32; o modo "
            "experimental nunca é usado nas gerações normais."
        )
        self.tts_cuda_attention_benchmark_button.clicked.connect(
            self._benchmark_tts_cuda_attention
        )

        hardware_actions = QHBoxLayout()
        hardware_actions.addWidget(self.tts_benchmark_button)
        hardware_actions.addWidget(QLabel("Gerações medidas por modo:"))
        hardware_actions.addWidget(self.tts_cuda_quant_benchmark_runs)
        hardware_actions.addWidget(self.tts_cuda_quant_benchmark_button)
        hardware_actions.addWidget(self.tts_cuda_fp16_benchmark_button)
        hardware_actions.addWidget(self.tts_cuda_tf32_benchmark_button)
        hardware_actions.addStretch(1)
        profile_actions = QHBoxLayout()
        profile_actions.addWidget(self.tts_cuda_profile_button)
        profile_actions.addStretch(1)
        attention_actions = QHBoxLayout()
        attention_actions.addWidget(self.tts_cuda_attention_benchmark_button)
        attention_actions.addStretch(1)
        hardware_form.addRow("Diagnóstico de gargalos", profile_actions)
        hardware_form.addRow("Backend de atenção", attention_actions)
        hardware_form.addRow("Perfil de desempenho", self.tts_performance_profile)
        hardware_form.addRow("Dispositivo para geração local", self.tts_device)
        hardware_form.addRow("Threads da CPU", self.tts_cpu_threads)
        hardware_form.addRow("Quantização na CPU", self.tts_quantization)
        hardware_form.addRow("Quantização na GPU", self.tts_cuda_quantization)
        hardware_form.addRow("Estado do hardware", self.tts_hardware_info)
        hardware_form.addRow("", hardware_actions)
        settings_layout.addWidget(hardware_group)
        self._refresh_tts_hardware_info()

        # Advanced generation controls are opt-in: disabled overrides preserve
        # Pocket TTS defaults, including automatic EOS handling and token limits.
        generation_group = QGroupBox("Parâmetros avançados do Pocket TTS")
        generation_form = QFormLayout(generation_group)

        self.tts_sampler_override = QCheckBox("Sobrescrever padrão do modelo")
        self.tts_sampler_override.setChecked(
            self.app_settings.value("tts_sampler_override", False, type=bool)
        )
        self.tts_sampler_value = QSpinBox()
        self.tts_sampler_value.setRange(1, 16)
        self.tts_sampler_value.setValue(
            self.app_settings.value("tts_sampler_value", 5, type=int)
        )
        self.tts_sampler_value.setEnabled(self.tts_sampler_override.isChecked())
        self.tts_sampler_value.setToolTip(
            "Mais passos podem melhorar a qualidade, mas também aumentam o tempo de síntese."
        )
        sampler_row = QWidget()
        sampler_layout = QHBoxLayout(sampler_row)
        sampler_layout.setContentsMargins(0, 0, 0, 0)
        sampler_layout.addWidget(self.tts_sampler_override)
        sampler_layout.addWidget(self.tts_sampler_value)
        sampler_layout.addStretch(1)
        generation_form.addRow("sampler_decode_steps", sampler_row)

        self.tts_temperature_override = QCheckBox("Sobrescrever padrão do modelo")
        self.tts_temperature_override.setChecked(
            self.app_settings.value("tts_temperature_override", False, type=bool)
        )
        self.tts_temperature_value = QDoubleSpinBox()
        self.tts_temperature_value.setRange(0.05, 2.0)
        self.tts_temperature_value.setDecimals(2)
        self.tts_temperature_value.setSingleStep(0.05)
        self.tts_temperature_value.setValue(
            self.app_settings.value("tts_temperature_value", 0.7, type=float)
        )
        self.tts_temperature_value.setEnabled(self.tts_temperature_override.isChecked())
        self.tts_temperature_value.setToolTip(
            "Temperatura altera a amostragem; não é um controle direto de velocidade."
        )
        temperature_row = QWidget()
        temperature_layout = QHBoxLayout(temperature_row)
        temperature_layout.setContentsMargins(0, 0, 0, 0)
        temperature_layout.addWidget(self.tts_temperature_override)
        temperature_layout.addWidget(self.tts_temperature_value)
        temperature_layout.addStretch(1)
        generation_form.addRow("temperature (temp)", temperature_row)

        self.tts_eos_threshold_override = QCheckBox("Sobrescrever padrão do modelo")
        self.tts_eos_threshold_override.setChecked(
            self.app_settings.value("tts_eos_threshold_override", False, type=bool)
        )
        self.tts_eos_threshold_value = QDoubleSpinBox()
        self.tts_eos_threshold_value.setRange(-20.0, 20.0)
        self.tts_eos_threshold_value.setDecimals(2)
        self.tts_eos_threshold_value.setSingleStep(0.1)
        self.tts_eos_threshold_value.setValue(
            self.app_settings.value("tts_eos_threshold_value", -4.0, type=float)
        )
        self.tts_eos_threshold_value.setEnabled(
            self.tts_eos_threshold_override.isChecked()
        )
        self.tts_eos_threshold_value.setToolTip(
            "Altera o limiar de fim de fala. Valores inadequados podem cortar palavras."
        )
        eos_row = QWidget()
        eos_layout = QHBoxLayout(eos_row)
        eos_layout.setContentsMargins(0, 0, 0, 0)
        eos_layout.addWidget(self.tts_eos_threshold_override)
        eos_layout.addWidget(self.tts_eos_threshold_value)
        eos_layout.addStretch(1)
        generation_form.addRow("eos_threshold", eos_row)

        self.tts_frames_after_eos = QSpinBox()
        self.tts_frames_after_eos.setRange(0, 20)
        self.tts_frames_after_eos.setValue(
            self.app_settings.value("tts_frames_after_eos", 0, type=int)
        )
        self.tts_frames_after_eos.setToolTip(
            "0 mantém a escolha automática do Pocket TTS. Valores menores que o padrão "
            "podem cortar o final da fala."
        )
        generation_form.addRow("frames_after_eos (0 = automático)", self.tts_frames_after_eos)

        self.tts_max_tokens = QSpinBox()
        self.tts_max_tokens.setRange(0, 10_000)
        self.tts_max_tokens.setValue(
            self.app_settings.value("tts_max_tokens", 0, type=int)
        )
        self.tts_max_tokens.setToolTip(
            "0 mantém o limite padrão do modelo; valores baixos podem truncar a geração."
        )
        generation_form.addRow("max_tokens (0 = padrão)", self.tts_max_tokens)

        self.tts_teacher_force_previous_chunk = QCheckBox(
            "Usar teacher forcing do bloco anterior"
        )
        self.tts_teacher_force_previous_chunk.setChecked(
            self.app_settings.value(
                "tts_teacher_force_previous_chunk", False, type=bool
            )
        )
        self.tts_teacher_force_previous_chunk.setToolTip(
            "Melhora a continuidade entre blocos em alguns cenários, mas adiciona prefill "
            "e pode deixar a geração mais lenta. Desligado por padrão."
        )
        generation_form.addRow("", self.tts_teacher_force_previous_chunk)

        for checkbox, spinbox in (
            (self.tts_sampler_override, self.tts_sampler_value),
            (self.tts_temperature_override, self.tts_temperature_value),
            (self.tts_eos_threshold_override, self.tts_eos_threshold_value),
        ):
            checkbox.toggled.connect(spinbox.setEnabled)
            checkbox.toggled.connect(self._save_tts_generation_parameters)
            spinbox.valueChanged.connect(self._save_tts_generation_parameters)
        self.tts_frames_after_eos.valueChanged.connect(
            self._save_tts_generation_parameters
        )
        self.tts_max_tokens.valueChanged.connect(self._save_tts_generation_parameters)
        self.tts_teacher_force_previous_chunk.toggled.connect(
            self._save_tts_generation_parameters
        )

        generation_note = QLabel(
            "As opções desativadas preservam os padrões vigentes de cada backend. "
            "0 em frames_after_eos e max_tokens significa automático/padrão. "
            "Teacher forcing fica desligado para priorizar velocidade."
        )
        generation_note.setObjectName("muted")
        generation_note.setWordWrap(True)
        generation_form.addRow("", generation_note)
        settings_layout.addWidget(generation_group)

        monitor_group = QGroupBox("Monitoramento em tempo real")
        monitor_form = QFormLayout(monitor_group)
        self.tts_monitor_cpu_ram = QLabel("Coletando métricas…")
        self.tts_monitor_gpu = QLabel("Consultando NVIDIA GPU…")
        self.tts_monitor_vram = QLabel("VRAM: —")
        for monitor_label in (
            self.tts_monitor_cpu_ram,
            self.tts_monitor_gpu,
            self.tts_monitor_vram,
        ):
            monitor_label.setWordWrap(True)
        monitor_form.addRow("CPU / RAM", self.tts_monitor_cpu_ram)
        monitor_form.addRow("GPU", self.tts_monitor_gpu)
        monitor_form.addRow("Memória da GPU", self.tts_monitor_vram)
        monitor_note = QLabel(
            "Atualização automática a cada 2,5 segundos. A leitura da NVIDIA usa nvidia-smi; "
            "a telemetria não limita a memória nem altera a configuração dos modelos."
        )
        monitor_note.setObjectName("muted")
        monitor_note.setWordWrap(True)
        monitor_form.addRow("", monitor_note)
        settings_layout.addWidget(monitor_group)

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

        self.clipboard_interactive_phrases = QCheckBox(
            "Ativar frases clicáveis no atalho (ouvir cada frase separadamente)"
        )
        self.clipboard_interactive_phrases.setToolTip(
            "Gera cada frase em um WAV separado dentro da sessão temporária, "
            "permitindo clicar nela para ouvir somente aquele trecho."
        )
        self.clipboard_interactive_phrases.setChecked(
            self.app_settings.value("clipboard_interactive_phrases", True, type=bool)
        )
        self.clipboard_interactive_phrases.toggled.connect(
            lambda checked: self.app_settings.setValue(
                "clipboard_interactive_phrases", bool(checked)
            )
        )

        self.clipboard_recognize_generated_audio = QCheckBox(
            "Reconhecer a fala após gerar (Parakeet) e sincronizar segmentos"
        )
        self.clipboard_recognize_generated_audio.setToolTip(
            "Após a geração, executa o reconhecimento de fala no WAV completo. "
            "Quando o backend fornece tempos, cada segmento reconhecido pode ser reproduzido "
            "isoladamente. Pode aumentar o tempo de espera e carregar o modelo ASR."
        )
        self.clipboard_recognize_generated_audio.setChecked(
            self.app_settings.value("clipboard_recognize_generated_audio", False, type=bool)
        )
        self.clipboard_recognize_generated_audio.toggled.connect(
            lambda checked: self.app_settings.setValue(
                "clipboard_recognize_generated_audio", bool(checked)
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
        output_form.addRow("", self.clipboard_interactive_phrases)
        output_form.addRow("", self.clipboard_recognize_generated_audio)
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
            "model_variant": str(self.tts_model_variant.currentData() or "portuguese"),
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
        model_index = self.tts_model_variant.findData(profile.get("model_variant", ""))
        if model_index >= 0:
            self.tts_model_variant.setCurrentIndex(model_index)
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
        tabs = self.centralWidget()
        if (
            hasattr(self, "history_panel")
            and isinstance(tabs, QTabWidget)
            and tabs.currentIndex() == 3
        ):
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
        model_variant: str = "",
        generation_settings: dict[str, object] | None = None,
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
                model_variant=model_variant,
                backend=backend,
                text_content=stored_text,
                generation_settings=generation_settings,
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
            model_variant=task.model_variant,
            backend="Google Colab" if task.remote else "Local",
            original_text=task.text,
            generation_settings={
                **build_generation_settings(
                    device=task.tts_device,
                    cpu_threads=task.cpu_threads,
                    quantization=task.tts_quantization,
                    profile=task.tts_profile,
                    remote=task.remote,
                ),
                "sampler_decode_steps": task.sampler_decode_steps,
                "temperature": task.temperature,
                "eos_threshold": task.eos_threshold,
                "frames_after_eos": task.frames_after_eos,
                "max_tokens": task.max_tokens,
                "teacher_force_previous_chunk": task.teacher_force_previous_chunk,
            },
        )

    def _record_clipboard_history(
        self,
        audio_path: str,
        performance: object,
        copied_text: str,
        language: str,
        voice_path: str | None,
        remote: bool,
        model_variant: str = "",
        generation_settings: dict[str, object] | None = None,
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
            model_variant=model_variant,
            generation_settings=generation_settings,
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
        QScrollArea#colabScrollArea {
            border: none;
            background: transparent;
        }
        QScrollArea#colabScrollArea QScrollBar:vertical {
            background: #20242d;
            width: 12px;
            margin: 2px;
            border-radius: 6px;
        }
        QScrollArea#colabScrollArea QScrollBar::handle:vertical {
            background: #56647a;
            min-height: 34px;
            border-radius: 6px;
        }
        QScrollArea#colabScrollArea QScrollBar::handle:vertical:hover {
            background: #7185a4;
        }
        QScrollArea#colabScrollArea QScrollBar::add-line:vertical,
        QScrollArea#colabScrollArea QScrollBar::sub-line:vertical {
            height: 0;
        }
        QScrollArea#colabScrollArea QScrollBar::add-page:vertical,
        QScrollArea#colabScrollArea QScrollBar::sub-page:vertical {
            background: transparent;
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
        QTextEdit, QLineEdit, QComboBox, QSpinBox {
            background: #20242d;
            color: #f1f3f8;
            border: 1px solid #3a4150;
            border-radius: 7px;
            padding: 7px 9px;
            selection-background-color: #52678e;
        }
        QSpinBox {
            min-height: 24px;
        }
        QSpinBox:focus {
            border: 1px solid #7697d4;
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
            ("Neerlandês", "dutch"), ("Japonês", "japanese"),
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
            "japanese": "Japonês",
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


    def _refresh_tts_model_variants(self) -> None:
        language = str(self.tts_language.currentData() or "portuguese")
        preferred = str(
            self._tts_model_preferences.get(language, default_model_variant(language))
        )
        options = model_variants_for_language(language)
        self.tts_model_variant.blockSignals(True)
        self.tts_model_variant.clear()
        for variant, label, description in options:
            self.tts_model_variant.addItem(label, variant)
            self.tts_model_variant.setItemData(
                self.tts_model_variant.count() - 1,
                description,
                Qt.ItemDataRole.ToolTipRole,
            )
        selected = self.tts_model_variant.findData(preferred)
        if selected < 0:
            selected = self.tts_model_variant.findData(default_model_variant(language))
        self.tts_model_variant.setCurrentIndex(max(0, selected))
        self.tts_model_variant.blockSignals(False)
        self._save_tts_model_variant()

    def _save_tts_model_variant(self, *_args) -> None:
        if not hasattr(self, "tts_model_variant"):
            return
        language = str(self.tts_language.currentData() or "portuguese")
        variant = str(
            self.tts_model_variant.currentData() or default_model_variant(language)
        )
        description = ""
        for candidate, _label, details in model_variants_for_language(language):
            if candidate == variant:
                description = details
                break
        self.tts_model_variant_info.setText(description)
        self.tts_model_variant.setToolTip(description)
        self._tts_model_preferences[language] = variant
        self.app_settings.setValue("tts_model_variant", variant)
        self.app_settings.setValue(
            "tts_model_variants",
            json.dumps(self._tts_model_preferences, ensure_ascii=False, sort_keys=True),
        )

    def _on_tts_language_changed(self, *_args) -> None:
        language = str(self.tts_language.currentData() or "portuguese")
        self.app_settings.setValue("tts_language", language)
        self._refresh_tts_model_variants()

    def _apply_tts_performance_profile(self, *_args) -> None:
        profile = str(self.tts_performance_profile.currentData() or "custom")
        self.app_settings.setValue("tts_performance_profile", profile)
        if profile == "custom":
            return
        try:
            settings = performance_profile_settings(
                profile,
                cpu_count=os.cpu_count() or 1,
                cuda_available=bool(torch.cuda.is_available()),
            )
        except (RuntimeError, ValueError) as exc:
            QMessageBox.warning(
                self, "Perfil indisponível", f"Não foi possível aplicar o perfil: {exc}"
            )
            return

        self._applying_tts_profile = True
        try:
            device_index = self.tts_device.findData(settings["device"])
            if device_index >= 0:
                self.tts_device.setCurrentIndex(device_index)
            self.tts_cpu_threads.setValue(int(settings["cpu_threads"]))
            self.tts_quantization.setChecked(bool(settings["quantization"]))
        finally:
            self._applying_tts_profile = False

        self.app_settings.setValue("tts_device", str(self.tts_device.currentData() or "cpu"))
        self.app_settings.setValue("tts_cpu_threads", int(self.tts_cpu_threads.value()))
        self.app_settings.setValue(
            "tts_quantization", bool(self.tts_quantization.isChecked())
        )
        self.app_settings.setValue(
            "tts_cuda_quantization", bool(self.tts_cuda_quantization.isChecked())
        )
        self.app_settings.setValue("tts_performance_profile", profile)
        self._refresh_tts_hardware_info()
        self.statusBar().showMessage(
            f"Perfil {self.tts_performance_profile.currentText()} aplicado.", 4000
        )

    def _save_tts_hardware_settings(self, *_args) -> None:
        if not all(
            hasattr(self, name)
            for name in (
                "tts_device",
                "tts_cpu_threads",
                "tts_quantization",
                "tts_cuda_quantization",
            )
        ):
            return
        if not self._applying_tts_profile and hasattr(self, "tts_performance_profile"):
            self.tts_performance_profile.blockSignals(True)
            custom_index = self.tts_performance_profile.findData("custom")
            if custom_index >= 0:
                self.tts_performance_profile.setCurrentIndex(custom_index)
            self.tts_performance_profile.blockSignals(False)
            self.app_settings.setValue("tts_performance_profile", "custom")
        self.app_settings.setValue("tts_device", str(self.tts_device.currentData() or "cpu"))
        self.app_settings.setValue("tts_cpu_threads", int(self.tts_cpu_threads.value()))
        self.app_settings.setValue(
            "tts_quantization", bool(self.tts_quantization.isChecked())
        )
        self.app_settings.setValue(
            "tts_cuda_quantization", bool(self.tts_cuda_quantization.isChecked())
        )
        self._refresh_tts_hardware_info()

    def _tts_quantization_enabled_for_device(self, device: str | None = None) -> bool:
        """Select the explicit INT8 preference for the chosen local device."""
        selected = device or str(self.tts_device.currentData() or "cpu")
        if selected == "cuda":
            return bool(self.tts_cuda_quantization.isChecked())
        return bool(self.tts_quantization.isChecked())

    def _tts_generation_parameters(self) -> dict[str, object]:
        """Return optional overrides; None preserves the Pocket TTS model defaults."""
        return {
            "sampler_decode_steps": (
                int(self.tts_sampler_value.value())
                if self.tts_sampler_override.isChecked()
                else None
            ),
            "temperature": (
                float(self.tts_temperature_value.value())
                if self.tts_temperature_override.isChecked()
                else None
            ),
            "eos_threshold": (
                float(self.tts_eos_threshold_value.value())
                if self.tts_eos_threshold_override.isChecked()
                else None
            ),
            "frames_after_eos": (
                int(self.tts_frames_after_eos.value())
                if self.tts_frames_after_eos.value() > 0
                else None
            ),
            "max_tokens": (
                int(self.tts_max_tokens.value())
                if self.tts_max_tokens.value() > 0
                else None
            ),
            "teacher_force_previous_chunk": bool(
                self.tts_teacher_force_previous_chunk.isChecked()
            ),
        }

    def _save_tts_generation_parameters(self, *_args) -> None:
        if not hasattr(self, "tts_sampler_override"):
            return
        self.app_settings.setValue(
            "tts_sampler_override", self.tts_sampler_override.isChecked()
        )
        self.app_settings.setValue("tts_sampler_value", self.tts_sampler_value.value())
        self.app_settings.setValue(
            "tts_temperature_override", self.tts_temperature_override.isChecked()
        )
        self.app_settings.setValue(
            "tts_temperature_value", self.tts_temperature_value.value()
        )
        self.app_settings.setValue(
            "tts_eos_threshold_override", self.tts_eos_threshold_override.isChecked()
        )
        self.app_settings.setValue(
            "tts_eos_threshold_value", self.tts_eos_threshold_value.value()
        )
        self.app_settings.setValue(
            "tts_frames_after_eos", self.tts_frames_after_eos.value()
        )
        self.app_settings.setValue("tts_max_tokens", self.tts_max_tokens.value())
        self.app_settings.setValue(
            "tts_teacher_force_previous_chunk",
            self.tts_teacher_force_previous_chunk.isChecked(),
        )

    def _refresh_tts_hardware_info(self) -> None:
        if not hasattr(self, "tts_hardware_info"):
            return
        logical_cpus = os.cpu_count() or 1
        cuda_available = bool(torch.cuda.is_available())
        if cuda_available:
            try:
                gpu_name = torch.cuda.get_device_name(0)
                vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                cuda_status = f"CUDA disponível: {gpu_name} ({vram_gb:.1f} GB de VRAM)."
            except Exception as exc:  # noqa: BLE001
                cuda_status = f"CUDA detectada, mas não foi possível consultar a GPU: {exc}"
        else:
            cuda_status = (
                "CUDA indisponível nesta instalação do PyTorch. Para usar a NVIDIA GPU, "
                "instale uma build CUDA compatível com o driver."
            )
        use_cpu = str(self.tts_device.currentData() or "cpu") == "cpu"
        self.tts_quantization.setEnabled(use_cpu)
        self.tts_cuda_quantization.setEnabled(not use_cpu and cuda_available)
        if use_cpu:
            quantization_status = (
                "CPU: INT8 ativada."
                if self.tts_quantization.isChecked()
                else "CPU: inferência sem quantização."
            )
            self.tts_cuda_quantization.setToolTip(
                "Preferência salva para a próxima seleção de CUDA; desmarcada por padrão."
            )
        elif cuda_available:
            quantization_status = (
                "CUDA: INT8 TorchAO solicitada."
                if self.tts_cuda_quantization.isChecked()
                else "CUDA: inferência sem quantização (caminho atual preservado)."
            )
            self.tts_cuda_quantization.setToolTip(
                "Ativa INT8 via TorchAO nas camadas de atenção e FFN. "
                "Desmarcada, o modelo carrega e gera exatamente como antes, sem quantização."
            )
        else:
            quantization_status = (
                "CUDA selecionada, mas indisponível: quantização na GPU não será usada."
            )
            self.tts_cuda_quantization.setToolTip(
                "Ative uma build CUDA do PyTorch para habilitar INT8 na GPU."
            )
        self.tts_hardware_info.setText(
            f"CPU: {logical_cpus} processador(es) lógico(s). {cuda_status} "
            f"{quantization_status}"
        )

    def _refresh_hardware_metrics(self) -> None:
        if self.hardware_monitor_worker is not None:
            return
        worker = FunctionWorker(read_hardware_metrics)
        self.hardware_monitor_worker = worker

        def completed(snapshot: object) -> None:
            self.hardware_monitor_worker = None
            if not hasattr(snapshot, "cpu_percent"):
                return

            def percent(value: float | None) -> str:
                return "—" if value is None else f"{value:.0f}%".replace(".", ",")

            def gib(value: float | None) -> str:
                return "—" if value is None else f"{value / 1024:.1f} GB".replace(".", ",")

            cpu_label = percent(snapshot.cpu_percent)
            if snapshot.ram_used_mb is None or snapshot.ram_total_mb is None:
                ram_label = "RAM: —"
            else:
                ram_label = f"RAM: {gib(snapshot.ram_used_mb)} / {gib(snapshot.ram_total_mb)}"
            self.tts_monitor_cpu_ram.setText(f"CPU: {cpu_label}   •   {ram_label}")

            if snapshot.gpu_name:
                utilization = percent(snapshot.gpu_utilization_percent)
                temperature = (
                    "—"
                    if snapshot.gpu_temperature_c is None
                    else f"{snapshot.gpu_temperature_c:.0f} °C"
                )
                self.tts_monitor_gpu.setText(
                    f"{snapshot.gpu_name}   •   Uso: {utilization}   •   "
                    f"Temperatura: {temperature}"
                )
                if snapshot.vram_used_mb is None or snapshot.vram_total_mb is None:
                    self.tts_monitor_vram.setText("VRAM: —")
                else:
                    self.tts_monitor_vram.setText(
                        f"VRAM: {snapshot.vram_used_mb / 1024:.1f} GB / "
                        f"{snapshot.vram_total_mb / 1024:.1f} GB"
                    )
            else:
                self.tts_monitor_gpu.setText(
                    snapshot.gpu_error or "GPU NVIDIA não detectada."
                )
                self.tts_monitor_vram.setText("VRAM: indisponível")

        def failed(message: str) -> None:
            self.hardware_monitor_worker = None
            self.tts_monitor_gpu.setText("Falha ao ler métricas da GPU.")
            self.tts_monitor_vram.setText("VRAM: indisponível")
            self.tts_monitor_cpu_ram.setToolTip(message)

        worker.signals.finished.connect(completed)
        worker.signals.failed.connect(failed)
        self.hardware_monitor_pool.start(worker)

    def _benchmark_tts_performance(self) -> None:
        if (
            self.active_worker is not None
            or self.active_generation_worker is not None
            or self.clipboard_generation_worker is not None
        ):
            QMessageBox.information(
                self,
                "Joe ocupado",
                "Aguarde a tarefa atual terminar antes de iniciar a comparação de desempenho.",
            )
            return

        language = str(self.tts_language.currentData() or "portuguese")
        model_variant = str(self.tts_model_variant.currentData() or language)
        cpu_threads = int(self.tts_cpu_threads.value())
        quantize_cpu = bool(self.tts_quantization.isChecked())
        quantize_cuda = bool(self.tts_cuda_quantization.isChecked())
        selected_voice = self.voice_path.text().strip() or None
        source_text = self.tts_text.toPlainText().strip()
        if source_text:
            try:
                prepared_text = self._prepared_speech_text(source_text, language)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Não foi possível preparar o texto",
                    f"Revise as opções de pronúncia antes de comparar o desempenho.\n\n{exc}",
                )
                return
            chunks = split_text_into_chunks(prepared_text, max_chars=280)
            sample_text = chunks[0] if chunks else ""
        else:
            sample_text = (
                "Este é um teste de desempenho do Pocket TTS para comparar "
                "a velocidade de geração usando o processador e a placa de vídeo."
            )

        def benchmark() -> str:
            results = self.tts.benchmark_devices(
                sample_text,
                language,
                selected_voice,
                model_variant,
                cpu_threads=cpu_threads,
                repeats=2,
                quantize=quantize_cpu,
                quantize_cuda=quantize_cuda,
            )
            return format_tts_benchmark_report(results, cpu_threads=cpu_threads)

        def completed(report: object) -> None:
            QMessageBox.information(
                self,
                "Comparação de desempenho — Pocket TTS",
                str(report),
            )

        self._run_task("Comparando o Pocket TTS em CPU e CUDA…", benchmark, completed)

    def _benchmark_tts_cuda_quantization(self) -> None:
        if (
            self.active_worker is not None
            or self.active_generation_worker is not None
            or self.clipboard_generation_worker is not None
        ):
            QMessageBox.information(
                self,
                "Joe ocupado",
                "Aguarde a tarefa atual terminar antes de comparar a quantização CUDA.",
            )
            return
        if not torch.cuda.is_available():
            QMessageBox.information(
                self,
                "CUDA indisponível",
                "Ative a GPU NVIDIA e confirme que o PyTorch reconhece CUDA antes de medir.",
            )
            return

        language = str(self.tts_language.currentData() or "portuguese")
        model_variant = str(self.tts_model_variant.currentData() or language)
        cpu_threads = int(self.tts_cpu_threads.value())
        measured_runs = int(self.tts_cuda_quant_benchmark_runs.value())
        selected_voice = self.voice_path.text().strip() or None
        source_text = self.tts_text.toPlainText().strip()
        if source_text:
            try:
                prepared_text = self._prepared_speech_text(source_text, language)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Não foi possível preparar o texto",
                    f"Revise as opções de pronúncia antes de comparar o desempenho.\n\n{exc}",
                )
                return
            chunks = split_text_into_chunks(prepared_text, max_chars=280)
            sample_text = chunks[0] if chunks else ""
        else:
            sample_text = (
                "Este é um teste repetido do Pocket TTS para medir a velocidade "
                "da GPU com quantização INT8 e sem quantização."
            )

        def benchmark() -> str:
            results = self.tts.benchmark_cuda_quantization(
                sample_text,
                language,
                selected_voice,
                model_variant,
                cpu_threads=cpu_threads,
                warmup_runs=2,
                measured_runs=measured_runs,
            )
            return format_cuda_quantization_benchmark_report(results)

        def completed(report: object) -> None:
            QMessageBox.information(
                self,
                "Medições CUDA — INT8 × sem INT8",
                str(report),
            )

        self._run_task(
            f"Medindo CUDA sem INT8 e com INT8: 2 aquecimentos + "
            f"{measured_runs} gerações por modo…",
            benchmark,
            completed,
        )

    def _benchmark_tts_cuda_precision(self) -> None:
        if (
            self.active_worker is not None
            or self.active_generation_worker is not None
            or self.clipboard_generation_worker is not None
        ):
            QMessageBox.information(
                self,
                "Joe ocupado",
                "Aguarde a tarefa atual terminar antes de comparar FP32 e FP16 na CUDA.",
            )
            return
        if not torch.cuda.is_available():
            QMessageBox.information(
                self,
                "CUDA indisponível",
                "Ative a GPU NVIDIA e confirme que o PyTorch reconhece CUDA antes de medir.",
            )
            return

        language = str(self.tts_language.currentData() or "portuguese")
        model_variant = str(self.tts_model_variant.currentData() or language)
        cpu_threads = int(self.tts_cpu_threads.value())
        measured_runs = int(self.tts_cuda_quant_benchmark_runs.value())
        selected_voice = self.voice_path.text().strip() or None
        source_text = self.tts_text.toPlainText().strip()
        if source_text:
            try:
                prepared_text = self._prepared_speech_text(source_text, language)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Não foi possível preparar o texto",
                    f"Revise as opções de pronúncia antes de comparar o desempenho.\n\n{exc}",
                )
                return
            chunks = split_text_into_chunks(prepared_text, max_chars=280)
            sample_text = chunks[0] if chunks else ""
        else:
            sample_text = (
                "Este é um teste repetido do Pocket TTS para comparar a precisão "
                "FP32 com a precisão mista FP16 na GPU NVIDIA."
            )

        def benchmark() -> str:
            results = self.tts.benchmark_cuda_precision(
                sample_text,
                language,
                selected_voice,
                model_variant,
                cpu_threads=cpu_threads,
                warmup_runs=2,
                measured_runs=measured_runs,
            )
            return format_cuda_precision_benchmark_report(results)

        def completed(report: object) -> None:
            QMessageBox.information(
                self,
                "Medições CUDA — FP32 × FP16",
                str(report),
            )

        self._run_task(
            f"Medindo CUDA FP32 e FP16 autocast: 2 aquecimentos + "
            f"{measured_runs} gerações por modo…",
            benchmark,
            completed,
        )


    def _benchmark_tts_cuda_tf32(self) -> None:
        if (
            self.active_worker is not None
            or self.active_generation_worker is not None
            or self.clipboard_generation_worker is not None
        ):
            QMessageBox.information(
                self,
                "Joe ocupado",
                "Aguarde a tarefa atual terminar antes de comparar FP32 IEEE e TF32 na CUDA.",
            )
            return
        if not torch.cuda.is_available():
            QMessageBox.information(
                self,
                "CUDA indisponível",
                "Ative a GPU NVIDIA e confirme que o PyTorch reconhece CUDA antes de medir.",
            )
            return

        language = str(self.tts_language.currentData() or "portuguese")
        model_variant = str(self.tts_model_variant.currentData() or language)
        cpu_threads = int(self.tts_cpu_threads.value())
        measured_runs = int(self.tts_cuda_quant_benchmark_runs.value())
        selected_voice = self.voice_path.text().strip() or None
        source_text = self.tts_text.toPlainText().strip()
        if source_text:
            try:
                prepared_text = self._prepared_speech_text(source_text, language)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Não foi possível preparar o texto",
                    f"Revise as opções de pronúncia antes de comparar o desempenho.\\n\\n{exc}",
                )
                return
            chunks = split_text_into_chunks(prepared_text, max_chars=280)
            sample_text = chunks[0] if chunks else ""
        else:
            sample_text = (
                "Este é um teste repetido do Pocket TTS para comparar "
                "FP32 IEEE com multiplicações TF32 na GPU NVIDIA."
            )

        def benchmark() -> str:
            results = self.tts.benchmark_cuda_tf32(
                sample_text,
                language,
                selected_voice,
                model_variant,
                cpu_threads=cpu_threads,
                warmup_runs=2,
                measured_runs=measured_runs,
            )
            return format_cuda_tf32_benchmark_report(results)

        def completed(report: object) -> None:
            QMessageBox.information(
                self,
                "Medições CUDA — FP32 IEEE × TF32",
                str(report),
            )

        self._run_task(
            f"Medindo CUDA FP32 IEEE e TF32: 2 aquecimentos + "
            f"{measured_runs} gerações por modo…",
            benchmark,
            completed,
        )

    def _benchmark_tts_cuda_attention(self) -> None:
        if (
            self.active_worker is not None
            or self.active_generation_worker is not None
            or self.clipboard_generation_worker is not None
        ):
            QMessageBox.information(
                self,
                "Joe ocupado",
                "Aguarde a tarefa atual terminar antes de comparar os backends de atenção CUDA.",
            )
            return
        if not torch.cuda.is_available():
            QMessageBox.information(
                self,
                "CUDA indisponível",
                "Ative a GPU NVIDIA e confirme que o PyTorch reconhece CUDA antes de medir.",
            )
            return

        language = str(self.tts_language.currentData() or "portuguese")
        model_variant = str(self.tts_model_variant.currentData() or language)
        cpu_threads = int(self.tts_cpu_threads.value())
        measured_runs = int(self.tts_cuda_quant_benchmark_runs.value())
        selected_voice = self.voice_path.text().strip() or None
        source_text = self.tts_text.toPlainText().strip()
        if source_text:
            try:
                prepared_text = self._prepared_speech_text(source_text, language)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Não foi possível preparar o texto",
                    f"Revise as opções de pronúncia antes de comparar a atenção.\n\n{exc}",
                )
                return
            chunks = split_text_into_chunks(prepared_text, max_chars=280)
            sample_text = chunks[0] if chunks else ""
        else:
            sample_text = (
                "Este teste compara a atenção SDPA padrão em FP32 com Flash Attention "
                "em FP16 restrito à operação de atenção."
            )

        def benchmark() -> str:
            results = self.tts.benchmark_cuda_attention(
                sample_text,
                language,
                selected_voice,
                model_variant,
                cpu_threads=cpu_threads,
                warmup_runs=2,
                measured_runs=measured_runs,
            )
            return format_cuda_attention_benchmark_report(results)

        def completed(report: object) -> None:
            QMessageBox.information(
                self,
                "Comparação CUDA — SDPA FP32 × backends fused FP16",
                str(report),
            )

        self._run_task(
            f"Comparando SDPA FP32 com backends fused FP16: 2 aquecimentos + "
            f"{measured_runs} gerações por modo…",
            benchmark,
            completed,
        )

    def _benchmark_tts_cuda_profile(self) -> None:
        if (
            self.active_worker is not None
            or self.active_generation_worker is not None
            or self.clipboard_generation_worker is not None
        ):
            QMessageBox.information(
                self,
                "Joe ocupado",
                "Aguarde a tarefa atual terminar antes de iniciar o diagnóstico CUDA.",
            )
            return
        if not torch.cuda.is_available():
            QMessageBox.information(
                self,
                "CUDA indisponível",
                "Ative a GPU NVIDIA e confirme que o PyTorch reconhece CUDA antes de diagnosticar.",
            )
            return

        language = str(self.tts_language.currentData() or "portuguese")
        model_variant = str(self.tts_model_variant.currentData() or language)
        cpu_threads = int(self.tts_cpu_threads.value())
        measured_runs = int(self.tts_cuda_quant_benchmark_runs.value())
        selected_voice = self.voice_path.text().strip() or None
        source_text = self.tts_text.toPlainText().strip()
        if source_text:
            try:
                prepared_text = self._prepared_speech_text(source_text, language)
            except Exception as exc:  # noqa: BLE001
                QMessageBox.warning(
                    self,
                    "Não foi possível preparar o texto",
                    f"Revise as opções de pronúncia antes do diagnóstico.\n\n{exc}",
                )
                return
            chunks = split_text_into_chunks(prepared_text, max_chars=280)
            sample_text = chunks[0] if chunks else ""
        else:
            sample_text = (
                "Este é um diagnóstico de desempenho do Pocket TTS para separar "
                "o custo da geração de latentes e da decodificação do áudio."
            )

        def benchmark() -> str:
            results = self.tts.benchmark_cuda_profile(
                sample_text,
                language,
                selected_voice,
                model_variant,
                cpu_threads=cpu_threads,
                warmup_runs=2,
                measured_runs=measured_runs,
            )
            return format_cuda_profile_benchmark_report(results)

        def completed(report: object) -> None:
            QMessageBox.information(
                self,
                "Diagnóstico CUDA — Flow LM × Mimi",
                str(report),
            )

        self._run_task(
            f"Diagnosticando CUDA FP32: 2 aquecimentos + "
            f"{measured_runs} gerações medidas…",
            benchmark,
            completed,
        )

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
        model_variant = str(self.tts_model_variant.currentData() or language)

        def export() -> Path:
            return self.tts.export_voice_state(
                source,
                destination,
                language,
                model_variant,
                device_preference=str(self.tts_device.currentData() or "cpu"),
                cpu_threads=int(self.tts_cpu_threads.value()),
                quantize=self._tts_quantization_enabled_for_device(),
            )

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
        elif command.startswith("open-project:"):
            project_path = command[len("open-project:"):]
            if project_path:
                self._open_joe_audio_project(project_path)
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

    def _on_clipboard_voice_source_changed(self, _value: str = "") -> None:
        """Invalidate the shortcut's voice snapshot when the selected voice changes."""
        self.clipboard_voice_cache.invalidate()
        if self.clipboard_generation_worker is None:
            self.clipboard_voice_cache.cleanup_retired()

    def _clipboard_dialog_closed(self, dialog: ClipboardAudioDialog) -> None:
        if self.clipboard_audio_window is dialog:
            self.clipboard_audio_window = None
        generation_worker = self.clipboard_generation_worker
        if generation_worker is not None:
            generation_worker.cancel()
        elif self.clipboard_recognition_worker is None:
            self._remove_clipboard_temp_dir(dialog.temp_dir)
            if self.clipboard_temp_dir == dialog.temp_dir:
                self.clipboard_temp_dir = None
            dialog.temp_dir = None
        if (
            self.clipboard_only_mode
            and not self._replacing_clipboard_dialog
            and generation_worker is None
            and self.clipboard_recognition_worker is None
        ):
            QTimer.singleShot(0, QApplication.quit)

    def _start_clipboard_recognition(
        self,
        audio_path: Path,
        remote: ColabAPIClient | None,
        dialog: ClipboardAudioDialog,
        dialog_state: dict[str, bool],
        temp_dir: Path,
    ) -> None:
        if not audio_path.is_file():
            dialog.set_recognition_failed("O WAV completo não foi encontrado.")
            return

        if remote is not None:
            recognize = lambda: remote.transcribe(audio_path, timestamps=True)
        else:
            device = str(self.asr_device.currentData() or "cpu")
            recognize = lambda: self.asr.transcribe(
                audio_path,
                timestamps=True,
                device_preference=device,
            )

        worker = FunctionWorker(recognize)
        self.clipboard_recognition_worker = worker
        dialog.set_recognition_started()

        def cleanup_if_closed() -> None:
            if dialog_state["closed"]:
                self._remove_clipboard_temp_dir(temp_dir)
                if self.clipboard_temp_dir == temp_dir:
                    self.clipboard_temp_dir = None

        def completed(result: object) -> None:
            self.clipboard_recognition_worker = None
            if dialog_state["closed"]:
                cleanup_if_closed()
            else:
                dialog.set_recognition_result(result)
            self.statusBar().showMessage(
                "Reconhecimento do áudio do atalho concluído.", 5000
            )
            if self.clipboard_only_mode:
                QTimer.singleShot(0, QApplication.quit)

        def failed(message: str) -> None:
            self.clipboard_recognition_worker = None
            if dialog_state["closed"]:
                cleanup_if_closed()
            else:
                dialog.set_recognition_failed(message)
            detail = message.split("\n\n", 1)[0]
            self.statusBar().showMessage(
                f"Reconhecimento do áudio do atalho falhou: {detail}", 8000
            )
            if self.clipboard_only_mode:
                QTimer.singleShot(0, QApplication.quit)

        worker.signals.finished.connect(completed)
        worker.signals.failed.connect(failed)
        self.thread_pool.start(worker)

    def _open_joe_audio_project(self, filename: str | Path | None = None) -> None:
        """Open a saved .joeaudio project without regenerating its speech."""
        if isinstance(filename, bool):
            # QAction.triggered supplies a checked boolean when connected directly.
            filename = None

        if self.clipboard_audio_window is not None:
            self.statusBar().showMessage(
                "Feche a janela de áudio atual antes de abrir outro projeto.",
                5000,
            )
            self.clipboard_audio_window.raise_()
            self.clipboard_audio_window.activateWindow()
            return

        if filename is None:
            filename, _ = QFileDialog.getOpenFileName(
                self,
                "Abrir projeto Joe",
                str(Path.home()),
                f"Projeto Joe Audio (*{JOE_AUDIO_PROJECT_EXTENSION});;Todos os arquivos (*)",
            )
            if not filename:
                return
        else:
            filename = str(Path(filename).expanduser())

        temp_dir = Path(tempfile.mkdtemp(prefix="joe_project_"))
        try:
            project = load_joe_audio_project(filename, temp_dir)
            phrase_texts_by_index = project["phrase_texts"]
            phrase_texts = [
                phrase_texts_by_index[index]
                for index in sorted(phrase_texts_by_index)
            ]
            dialog = ClipboardAudioDialog(
                self.player,
                self.tts_output_directory.text().strip(),
                self,
                phrases=phrase_texts,
                recognition_enabled=bool(
                    project.get("recognized_text")
                    or project.get("recognized_segments")
                ),
            )
            dialog.setWindowTitle(f"Joe — {Path(filename).name}")
            dialog.temp_dir = temp_dir
            dialog.set_text_preview(str(project["original_text"]))
            phrase_paths = project["phrase_audio_paths"]
            for index in sorted(phrase_texts_by_index):
                phrase_path = phrase_paths.get(index)
                if phrase_path is not None:
                    dialog.add_chunk(
                        index,
                        phrase_path,
                        len(phrase_texts_by_index),
                        phrase_texts_by_index[index],
                    )
            dialog.set_ready({"path": str(project["full_audio_path"])})
            recognized_text = str(project.get("recognized_text") or "")
            recognized_segments = project.get("recognized_segments") or []
            if recognized_text or recognized_segments:
                dialog.set_recognition_result({
                    "text": recognized_text,
                    "segments": recognized_segments,
                    "backend": "Projeto salvo",
                })
                dialog.recognition_toggle.setVisible(True)
                dialog.recognition_toggle.setChecked(True)
            dialog.closed.connect(lambda: self._clipboard_dialog_closed(dialog))
            self.clipboard_audio_window = dialog
            self.clipboard_temp_dir = temp_dir
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
            self.statusBar().showMessage(
                f"Projeto aberto: {Path(filename).name}", 5000
            )
        except (OSError, ValueError, TypeError, KeyError, zipfile.BadZipFile) as exc:
            self._remove_clipboard_temp_dir(temp_dir)
            QMessageBox.critical(
                self,
                "Não foi possível abrir o projeto",
                f"O arquivo .joeaudio não pôde ser carregado.\\n\\n{exc}",
            )

    def _generate_audio_from_clipboard(self) -> None:
        shortcut_action = decide_clipboard_shortcut_action(
            dialog_open=self.clipboard_audio_window is not None,
            generation_active=(
                self.clipboard_generation_worker is not None
                or self.clipboard_recognition_worker is not None
                or self.active_worker is not None
            ),
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
            self.statusBar().showMessage(
                "Já existe uma tarefa em andamento. Aguarde a conclusão antes de iniciar outro áudio.",
                5000,
            )
            return

        if shortcut_action == "busy":
            self.statusBar().showMessage(
                "Já existe uma tarefa em andamento. Aguarde a conclusão antes de iniciar outro áudio.",
                4000,
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
        model_variant = str(self.tts_model_variant.currentData() or language)
        try:
            speech_text = self._prepared_speech_text(copied_text, language)
            remote = (
                self._colab_client()
                if self.processing_mode.currentData() == "colab"
                else None
            )
            chunk_size = int(self.clipboard_chunk_size.currentData())
            interactive_phrases = bool(
                self.clipboard_interactive_phrases.isChecked()
            )
            recognition_enabled = bool(
                self.clipboard_recognize_generated_audio.isChecked()
            )
            chunks = (
                split_text_into_phrases(speech_text, max_chars=chunk_size)
                if interactive_phrases
                else split_text_into_chunks(speech_text, max_chars=chunk_size)
            )
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
        voice_source = self.voice_path.text().strip() or None
        try:
            local_voice_path, remote_reference_audio = resolve_shortcut_voice_paths(
                self.clipboard_voice_cache,
                voice_source,
                remote=remote is not None,
            )
            voice = str(local_voice_path) if local_voice_path is not None else None
        except OSError as exc:
            self._remove_clipboard_temp_dir(temp_dir)
            self.statusBar().showMessage(
                f"Não foi possível preparar a voz temporária do atalho: {exc}",
                8000,
            )
            if self.clipboard_only_mode:
                QTimer.singleShot(0, QApplication.quit)
            return

        tts_device = str(self.tts_device.currentData() or "cpu")
        cpu_threads = int(self.tts_cpu_threads.value())
        tts_quantization = (
            bool(self.colab_tts_quantize_int8.isChecked())
            if remote is not None
            else self._tts_quantization_enabled_for_device(tts_device)
        )
        tts_generation_parameters = self._tts_generation_parameters()
        generation_settings = build_generation_settings(
            device=tts_device,
            cpu_threads=cpu_threads,
            quantization=tts_quantization,
            profile=str(self.tts_performance_profile.currentData() or "custom"),
            remote=remote is not None,
        )
        generation_settings.update(tts_generation_parameters)

        remote_voice_session = (
            ColabVoiceSession(
                remote,
                remote_reference_audio,
                language,
                model_variant,
                name=(Path(remote_reference_audio).stem if remote_reference_audio else None),
            )
            if remote is not None
            else None
        )

        def prepare_clipboard_voice() -> None:
            if remote_voice_session is not None:
                remote_voice_session.prepare()

        if remote_voice_session is not None:
            def generate_clipboard_chunk(chunk_text: str, output_path: Path) -> Path:
                audio_data = remote_voice_session.generate_audio(
                    chunk_text, **tts_generation_parameters
                )
                write_bytes_atomically(output_path, audio_data)
                return output_path
        else:
            def generate_clipboard_chunk(chunk_text: str, output_path: Path) -> Path:
                return self.tts.generate(
                    chunk_text,
                    language,
                    str(output_path),
                    voice,
                    model_variant,
                    device_preference=tts_device,
                    cpu_threads=cpu_threads,
                    quantize=self._tts_quantization_enabled_for_device(tts_device),
                    **tts_generation_parameters,
                )

        if shortcut_action == "replace":
            # Replace only after validating the new clipboard text and preparing
            # its settings, so an empty/invalid clipboard keeps the old result open.
            old_dialog = self.clipboard_audio_window
            if old_dialog is not None:
                self._replacing_clipboard_dialog = True
                try:
                    closed = old_dialog.close()
                except RuntimeError:
                    # The Qt object may already have been destroyed.
                    closed = True
                    if self.clipboard_audio_window is old_dialog:
                        self.clipboard_audio_window = None
                finally:
                    self._replacing_clipboard_dialog = False
                if not closed:
                    self._remove_clipboard_temp_dir(temp_dir)
                    self.statusBar().showMessage(
                        "Não foi possível substituir a janela de atalho aberta.",
                        5000,
                    )
                    return
                if self.clipboard_audio_window is old_dialog:
                    self.clipboard_audio_window = None

        dialog = ClipboardAudioDialog(
            self.player,
            self.tts_output_directory.text().strip(),
            self,
            phrases=chunks if interactive_phrases else [],
            recognition_enabled=recognition_enabled,
        )
        dialog.temp_dir = temp_dir
        dialog.set_text_preview(copied_text)
        dialog_state = {"closed": False}
        dialog.closed.connect(lambda: self._clipboard_dialog_closed(dialog))
        dialog.closed.connect(lambda: dialog_state.__setitem__("closed", True))
        dialog.saved.connect(
            lambda output_path, performance: self._record_clipboard_history(
                output_path,
                performance,
                copied_text,
                language,
                voice_source,
                remote is not None,
                model_variant,
                generation_settings,
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
            prepare=(
                prepare_clipboard_voice
                if remote_voice_session is not None and remote_reference_audio
                else None
            ),
        )
        self.clipboard_generation_worker = worker

        def chunk_ready(index: int, path: str, total: int, text: str) -> None:
            if not dialog_state["closed"]:
                dialog.add_chunk(index, Path(path), total, text)

        def progress(message: str) -> None:
            if not dialog_state["closed"]:
                dialog.set_progress(message)

        def cleanup_closed_dialog() -> None:
            self._remove_clipboard_temp_dir(temp_dir)
            if self.clipboard_temp_dir == temp_dir:
                self.clipboard_temp_dir = None
            if self.clipboard_only_mode and self.clipboard_recognition_worker is None:
                QTimer.singleShot(0, QApplication.quit)

        def finished(result: object) -> None:
            self.clipboard_generation_worker = None
            self.clipboard_voice_cache.cleanup_retired()
            result_data = result if isinstance(result, dict) else {"path": str(result)}
            if dialog_state["closed"]:
                cleanup_closed_dialog()
                return
            dialog.set_ready(result_data)
            self.statusBar().showMessage("Áudio do clipboard gerado e unido.", 5000)
            if recognition_enabled:
                self._start_clipboard_recognition(
                    destination,
                    remote,
                    dialog,
                    dialog_state,
                    temp_dir,
                )

        def failed(message: str) -> None:
            self.clipboard_generation_worker = None
            self.clipboard_voice_cache.cleanup_retired()
            parts = message.split("\n\n", 1)
            detail = parts[0].strip() or "Erro desconhecido"
            technical_details = parts[1] if len(parts) > 1 else ""
            if dialog_state["closed"]:
                cleanup_closed_dialog()
            else:
                dialog.set_failed(detail, technical_details)
                self.statusBar().showMessage(
                    f"Falha ao gerar áudio do clipboard: {detail}", 8000
                )

        def cancelled() -> None:
            self.clipboard_generation_worker = None
            self.clipboard_voice_cache.cleanup_retired()
            cleanup_closed_dialog()
            if self.clipboard_only_mode and self.clipboard_recognition_worker is None:
                QTimer.singleShot(0, QApplication.quit)

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
        model_variant = str(self.tts_model_variant.currentData() or language)
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
                model_variant=model_variant,
                tts_device=str(self.tts_device.currentData() or "cpu"),
                cpu_threads=int(self.tts_cpu_threads.value()),
                tts_quantization=(
                    bool(self.colab_tts_quantize_int8.isChecked())
                    if remote
                    else self._tts_quantization_enabled_for_device(
                        str(self.tts_device.currentData() or "cpu")
                    )
                ),
                tts_profile=str(self.tts_performance_profile.currentData() or "custom"),
                voice_path=selected_voice,
                voice_alias=selected_voice_alias,
                remote=remote,
                remote_url=remote_url,
                remote_token=remote_token,
                **self._tts_generation_parameters(),
            )
        except ValueError as exc:
            QMessageBox.information(self, "Tarefa inválida", str(exc))
            return

        self.statusBar().showMessage(f"Tarefa adicionada à fila: {task.output_name}", 4000)
        self._refresh_generation_queue()
        self._start_next_generation()

    def _open_generation_queue(self, selected_task_id: str | None = None) -> None:
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
        if selected_task_id:
            self.generation_queue_dialog.select_task(selected_task_id)
        self.generation_queue_dialog.show()
        self.generation_queue_dialog.raise_()
        self.generation_queue_dialog.activateWindow()

    def _refresh_generation_queue(self) -> None:
        active_count = sum(
            task.status in {"queued", "running", "cancelling"}
            for task in self.generation_queue.tasks()
        )
        failed_count = sum(
            task.status == "failed" for task in self.generation_queue.tasks()
        )
        suffix = f" · {failed_count} falha(s)" if failed_count else ""
        self.tts_queue_button.setText(
            f"Fila de geração ({active_count}){suffix}"
        )
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
        self.tts_speed_info.setText(
            "Geração falhou. A fila foi aberta com os detalhes técnicos do erro."
        )
        self._set_busy(False, "Falha na tarefa; continuando a fila")
        failed_task = self.generation_queue.get(task_id)
        task_name = failed_task.output_name if failed_task is not None else task_id
        error_summary = message.splitlines()[0] if message.splitlines() else "Erro desconhecido"
        self.statusBar().showMessage(
            f"Falha em {task_name}: {error_summary}",
            8000,
        )
        self._refresh_generation_queue()
        self._open_generation_queue(task_id)
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
        if hasattr(self, "advanced_player"):
            self.advanced_player.set_transcript(
                self.transcript_segments,
                self.transcript_audio_path if self.transcript_segments else None,
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
            if hasattr(self, "advanced_player"):
                self.advanced_player.set_transcript([], None)

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
            dialog = ErrorDetailsDialog("Falha no processamento", message, self)
            dialog.exec()

        worker.signals.finished.connect(finished)
        worker.signals.failed.connect(failed)
        self.thread_pool.start(worker)

    def _set_busy(self, busy: bool, message: str) -> None:
        for widget in (
            self.tts_generate_button,
            self.tts_play_button,
            self.tts_favorites_button,
            self.tts_settings_button,
            self.tts_device,
            self.tts_cpu_threads,
            self.tts_performance_profile,
            self.tts_quantization,
            self.tts_cuda_quantization,
            self.tts_sampler_override,
            self.tts_sampler_value,
            self.tts_temperature_override,
            self.tts_temperature_value,
            self.tts_eos_threshold_override,
            self.tts_eos_threshold_value,
            self.tts_frames_after_eos,
            self.tts_max_tokens,
            self.tts_teacher_force_previous_chunk,
            self.tts_benchmark_button,
            self.tts_cuda_quant_benchmark_runs,
            self.tts_cuda_quant_benchmark_button,
            self.tts_cuda_fp16_benchmark_button,
            self.tts_cuda_tf32_benchmark_button,
            self.tts_cuda_profile_button,
            self.tts_cuda_attention_benchmark_button,
            self.tts_language,
            self.tts_model_variant,
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
            self.colab_upload_model_button,
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
        self.hardware_monitor_timer.stop()
        self._stop_global_hotkey()
        if self.local_command_server is not None:
            self.local_command_server.stop()
            self.local_command_server = None
        if self.clipboard_audio_window is not None:
            self.clipboard_audio_window.close()
        if self.colab_monitor_pool.activeThreadCount():
            self.colab_monitor_pool.waitForDone(5000)
        if self.hardware_monitor_pool.activeThreadCount():
            self.hardware_monitor_pool.waitForDone(3000)
        if self.thread_pool.activeThreadCount():
            self.statusBar().showMessage("Aguardando a tarefa atual terminar…")
            self.thread_pool.waitForDone(-1)
        self._remove_clipboard_temp_dir(self.clipboard_temp_dir)
        self.clipboard_temp_dir = None
        self.clipboard_voice_cache.close()
        self.player.stop()
        self.asr.close()
        self.tts.close()
        super().closeEvent(event)


def main() -> None:
    arguments = list(sys.argv[1:])
    clipboard_requested = "--clipboard-audio" in arguments
    project_argument = next(
        (
            Path(argument).expanduser().resolve()
            for argument in arguments
            if Path(argument).suffix.lower() == JOE_AUDIO_PROJECT_EXTENSION
        ),
        None,
    )
    command = (
        "clipboard-audio"
        if clipboard_requested
        else f"open-project:{project_argument}"
        if project_argument is not None
        else "activate"
    )
    if send_local_command(command):
        return

    # Desktop launchers pass the selected file to the application, not to Qt.
    app = QApplication([sys.argv[0]])
    app.setApplicationName("Joe Audio Studio")
    app.setOrganizationName("g4b3r-desktop")
    icon_path = Path(__file__).resolve().parent / "resources" / "joe.png"
    if icon_path.is_file():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = JoeWindow()
    app.aboutToQuit.connect(window.clipboard_voice_cache.close)
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
        if project_argument is not None:
            QTimer.singleShot(
                0,
                lambda path=project_argument: window._open_joe_audio_project(path),
            )
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
