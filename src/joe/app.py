from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import traceback
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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from joe.local_ipc import LocalCommandServer, send_local_command
from joe.services.colab_api import ColabAPIClient
from joe.services.parakeet import ParakeetReduxService
from joe.services.pocket_tts import PocketTTSService
from joe.text_normalization import normalize_text_for_tts
from joe.utils import build_srt, build_audio_output_path


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


class HotkeyBridge(QObject):
    activated = Signal()
    command_received = Signal(str)


class ClipboardAudioDialog(QDialog):
    """Compact playback window for audio generated from the system clipboard."""

    closed = Signal()

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
        self.is_closing = False

        self.setWindowTitle("Joe — Texto copiado para áudio")
        self.setWindowFlags(self.windowFlags() | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(430, 250)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(10)

        self.preview_label = QLabel("Texto copiado")
        self.preview_label.setWordWrap(True)
        self.preview_label.setObjectName("muted")
        layout.addWidget(self.preview_label)

        self.status_label = QLabel("Gerando áudio…")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

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

    def set_text_preview(self, text: str) -> None:
        compact = " ".join(text.split())
        if len(compact) > 260:
            compact = compact[:257].rstrip() + "…"
        self.preview_label.setText(compact)

    def set_ready(self, audio_path: Path) -> None:
        self.audio_path = audio_path
        self.status_label.setText("Áudio pronto. Você pode reproduzir ou salvar uma cópia.")
        self.play_button.setEnabled(True)
        self.save_button.setEnabled(True)

    def set_failed(self, message: str) -> None:
        self.status_label.setText(f"Não foi possível gerar o áudio: {message}")
        self.play_button.setEnabled(False)
        self.pause_button.setEnabled(False)
        self.save_button.setEnabled(False)

    def _play(self) -> None:
        if not self.audio_path or not self.audio_path.is_file():
            return
        self.player.setSource(QUrl.fromLocalFile(str(self.audio_path.resolve())))
        self.player.play()
        self.pause_button.setEnabled(True)

    def _pause(self) -> None:
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

    def closeEvent(self, event) -> None:
        self.is_closing = True
        if self.audio_path:
            current_source = self.player.source().toLocalFile()
            if current_source == str(self.audio_path.resolve()):
                self.player.stop()
                self.player.setSource(QUrl())
        self.closed.emit()
        super().closeEvent(event)


class JoeWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Joe — Audio Studio")
        self.resize(980, 740)
        self.setMinimumSize(760, 600)

        self.tts = PocketTTSService()
        self.asr = ParakeetReduxService()
        self.app_settings = QSettings("g4b3r-desktop", "JoeAudioStudio")
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(1)
        self.active_worker: FunctionWorker | None = None
        self.clipboard_generation_worker: FunctionWorker | None = None
        self.clipboard_audio_window: ClipboardAudioDialog | None = None
        self.clipboard_temp_dir: Path | None = None
        self.hotkey_listener = None
        self.local_command_server: LocalCommandServer | None = None
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
        self.transcript_segments: list[dict] = []
        self.transcription_backend = ""
        self.pronunciation_dictionary = self._load_pronunciation_dictionary()

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)

        self._build_ui()
        self.setStyleSheet(self._stylesheet())
        self.statusBar().showMessage("Pronto — os modelos serão carregados sob demanda.")
        self._start_global_hotkey()
        self._update_colab_monitoring()

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_tts_tab(), "Texto → Áudio")
        tabs.addTab(self._build_asr_tab(), "Áudio → Texto")
        tabs.addTab(self._build_colab_tab(), "Google Colab / API")
        tabs.setCurrentIndex(self.app_settings.value("last_tab", 0, type=int))
        tabs.currentChanged.connect(
            lambda index: self.app_settings.setValue("last_tab", index)
        )
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
            "blob/feature/colab-model-api/colab/Joe_Colab_API.ipynb"
        )
        QDesktopServices.openUrl(QUrl(url))

    def _build_tts_tab(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(14)

        title = QLabel("Transforme texto em fala")
        title.setObjectName("pageTitle")
        description = QLabel(
            "Gere WAV com o Kyutai Pocket TTS. Atalho global: "
            "Ctrl+Shift+Espaço transforma o texto copiado em áudio imediatamente."
        )
        description.setObjectName("muted")

        self.tts_text = QTextEdit()
        self.tts_text.setPlaceholderText(
            "Digite ou cole aqui o texto que deseja transformar em áudio..."
        )
        self.tts_text.setPlainText(str(self.app_settings.value("tts_text", "")))
        self.tts_text.setMinimumHeight(240)
        self.tts_text.textChanged.connect(
            lambda: self._save_text_setting("tts_text", self.tts_text.toPlainText())
        )

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

        speech_tools_row = QHBoxLayout()
        speech_tools_row.addWidget(self.tts_normalization_enabled, 1)
        speech_tools_row.addWidget(self.tts_dictionary_button)
        speech_tools_row.addWidget(self.tts_preview_button)

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

        self.voice_path = QLineEdit(
            str(self.app_settings.value("voice_path", ""))
        )
        self.voice_path.setReadOnly(True)
        self.voice_path.setPlaceholderText("Voz padrão do Pocket TTS")
        self.voice_path.textChanged.connect(
            lambda value: self._save_text_setting("voice_path", value)
        )
        self.voice_button = QPushButton("Escolher áudio de referência…")
        self.voice_button.clicked.connect(self._choose_voice)

        options = QFormLayout()
        options.addRow("Pasta para salvar", output_folder_row)
        options.addRow("", self.tts_auto_filename)
        options.addRow("Idioma da fala", self.tts_language)
        voice_row = QWidget()
        voice_layout = QHBoxLayout(voice_row)
        voice_layout.setContentsMargins(0, 0, 0, 0)
        voice_layout.addWidget(self.voice_path, 1)
        voice_layout.addWidget(self.voice_button)
        options.addRow("Voz personalizada (opcional)", voice_row)

        self.tts_generate_button = QPushButton("Gerar áudio WAV")
        self.tts_generate_button.setObjectName("primaryButton")
        self.tts_generate_button.clicked.connect(self._generate_audio)

        self.tts_play_button = QPushButton("Reproduzir último áudio")
        self.tts_play_button.setEnabled(False)
        self.tts_play_button.clicked.connect(self._play_generated_audio)

        action_row = QHBoxLayout()
        action_row.addWidget(self.tts_generate_button)
        action_row.addWidget(self.tts_play_button)
        action_row.addStretch(1)

        self.tts_info = QLabel(
            "A normalização prepara uma cópia do texto para o Pocket TTS. "
            "O texto original permanece intacto na caixa acima. Você pode cadastrar "
            "pronúncias próprias no dicionário."
        )
        self.tts_info.setObjectName("muted")
        self.tts_info.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(self.tts_text, 1)
        layout.addLayout(speech_tools_row)
        layout.addLayout(options)
        layout.addLayout(action_row)
        layout.addWidget(self.tts_info)
        return page

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

        self.copy_button = QPushButton("Copiar texto")
        self.copy_button.clicked.connect(self._copy_transcript)
        self.save_txt_button = QPushButton("Salvar TXT…")
        self.save_txt_button.clicked.connect(self._save_transcript)
        self.save_srt_button = QPushButton("Exportar SRT…")
        self.save_srt_button.clicked.connect(self._save_srt)
        self.save_srt_button.setEnabled(False)

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
            "Cadastre uma palavra ou expressão e a forma como o TTS deve pronunciá-la. "
            "A correspondência não diferencia maiúsculas de minúsculas."
        )
        help_text.setWordWrap(True)
        layout.addWidget(help_text)

        table = QTableWidget(0, 2)
        table.setHorizontalHeaderLabels(["Palavra ou expressão", "Pronúncia desejada"])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        for term, pronunciation in sorted(self.pronunciation_dictionary.items()):
            row = table.rowCount()
            table.insertRow(row)
            table.setItem(row, 0, QTableWidgetItem(term))
            table.setItem(row, 1, QTableWidgetItem(pronunciation))
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
            updated: dict[str, str] = {}
            for row in range(table.rowCount()):
                term_item = table.item(row, 0)
                pronunciation_item = table.item(row, 1)
                if term_item and pronunciation_item:
                    term = term_item.text().strip()
                    pronunciation = pronunciation_item.text().strip()
                    if term and pronunciation:
                        updated[term] = pronunciation
            self._save_pronunciation_dictionary(updated)

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

    def _choose_voice(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Selecionar áudio de referência de voz",
            str(Path.home()),
            "Áudio (*.wav *.mp3 *.flac *.m4a);;Todos os arquivos (*)",
        )
        if path:
            self.voice_path.setText(path)

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
        if self.clipboard_generation_worker is None:
            self._remove_clipboard_temp_dir(dialog.temp_dir)
            if self.clipboard_temp_dir == dialog.temp_dir:
                self.clipboard_temp_dir = None
            dialog.temp_dir = None

    def _generate_audio_from_clipboard(self) -> None:
        if self.clipboard_generation_worker is not None:
            if self.clipboard_audio_window is not None:
                self.clipboard_audio_window.raise_()
                self.clipboard_audio_window.activateWindow()
            self.statusBar().showMessage("A geração do texto copiado já está em andamento.", 4000)
            return

        if self.active_worker is not None:
            self.statusBar().showMessage(
                "Aguarde a tarefa atual terminar antes de gerar outro áudio.", 5000
            )
            return

        copied_text = QApplication.clipboard().text().strip()
        if not copied_text:
            self.statusBar().showMessage(
                "O clipboard está vazio. Copie um texto e pressione Ctrl+Shift+Espaço.",
                6000,
            )
            return

        if self.clipboard_audio_window is not None:
            self.clipboard_audio_window.close()

        language = str(self.tts_language.currentData())
        try:
            speech_text = self._prepared_speech_text(copied_text, language)
            remote = (
                self._colab_client()
                if self.processing_mode.currentData() == "colab"
                else None
            )
        except Exception as exc:  # noqa: BLE001
            self.statusBar().showMessage(f"Não foi possível iniciar o TTS: {exc}", 8000)
            return

        try:
            temp_dir = Path(tempfile.mkdtemp(prefix="joe-clipboard-audio-"))
        except OSError as exc:
            self.statusBar().showMessage(f"Não foi possível criar áudio temporário: {exc}", 8000)
            return

        destination = temp_dir / "clipboard_audio.wav"
        voice = self.voice_path.text().strip() or None
        if remote is not None:
            def generate_clipboard_audio() -> Path:
                audio_data = remote.generate_audio(speech_text, language, voice)
                destination.write_bytes(audio_data)
                return destination
        else:
            def generate_clipboard_audio() -> Path:
                return self.tts.generate(
                    speech_text, language, str(destination), voice
                )

        dialog = ClipboardAudioDialog(
            self.player,
            self.tts_output_directory.text().strip(),
            self,
        )
        dialog.temp_dir = temp_dir
        dialog.set_text_preview(copied_text)
        dialog.closed.connect(lambda: self._clipboard_dialog_closed(dialog))
        self.clipboard_audio_window = dialog
        self.clipboard_temp_dir = temp_dir
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

        worker = FunctionWorker(generate_clipboard_audio)
        self.clipboard_generation_worker = worker

        def finished(result: object) -> None:
            self.clipboard_generation_worker = None
            if dialog.is_closing:
                self._remove_clipboard_temp_dir(temp_dir)
                dialog.temp_dir = None
                if self.clipboard_temp_dir == temp_dir:
                    self.clipboard_temp_dir = None
                return
            dialog.set_ready(Path(result))
            self.statusBar().showMessage("Áudio do clipboard gerado.", 5000)

        def failed(message: str) -> None:
            self.clipboard_generation_worker = None
            detail = message.split("\n\n", 1)[0].strip() or "Erro desconhecido"
            if dialog.is_closing:
                self._remove_clipboard_temp_dir(temp_dir)
                dialog.temp_dir = None
            else:
                dialog.set_failed(detail)
                self._remove_clipboard_temp_dir(temp_dir)
                dialog.temp_dir = None
            if self.clipboard_temp_dir == temp_dir:
                self.clipboard_temp_dir = None
            self.statusBar().showMessage(f"Falha ao gerar áudio do clipboard: {detail}", 8000)

        worker.signals.finished.connect(finished)
        worker.signals.failed.connect(failed)
        self.thread_pool.start(worker)

    def _generate_audio(self) -> None:
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
            except OSError as exc:
                QMessageBox.critical(
                    self, "Erro na pasta de saída", f"Não foi possível preparar a pasta:\n{exc}"
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
            self.tts_output_directory.setText(str(destination.parent))
            self._save_tts_output_settings()
            try:
                destination.parent.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                QMessageBox.critical(
                    self, "Erro na pasta de saída", f"Não foi possível criar a pasta:\n{exc}"
                )
                return

        language = str(self.tts_language.currentData())
        try:
            speech_text = self._prepared_speech_text(text, language)
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "Erro na normalização", str(exc))
            return
        voice = self.voice_path.text().strip() or None
        if self.processing_mode.currentData() == "colab":
            try:
                remote = self._colab_client()
            except ValueError as exc:
                QMessageBox.information(self, "Configure a API do Colab", str(exc))
                return

            def generate_remote() -> Path:
                audio_data = remote.generate_audio(speech_text, language, voice)
                destination.write_bytes(audio_data)
                return destination

            task = generate_remote
            status = "Gerando áudio no Google Colab…"
        else:
            task = lambda: self.tts.generate(speech_text, language, str(destination), voice)
            status = "Gerando áudio com Pocket TTS local…"

        self._run_task(status, task, self._audio_generated)

    def _audio_generated(self, result: object) -> None:
        self.generated_audio = Path(result)
        self.tts_play_button.setEnabled(True)
        self.tts_info.setText(f"Áudio salvo em: {self.generated_audio}")

    def _play_generated_audio(self) -> None:
        if not self.generated_audio or not self.generated_audio.is_file():
            return
        if self.player.source().isValid() and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.stop()
            return
        self.player.setSource(QUrl.fromLocalFile(str(self.generated_audio.resolve())))
        self.player.play()

    def _transcribe_audio(self) -> None:
        source = self.audio_path.text().strip()
        if not source:
            QMessageBox.information(self, "Áudio não selecionado", "Selecione um arquivo de áudio.")
            return

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
        self.transcript_segments = [
            segment for segment in list(data.get("segments", []))
            if str(segment.get("text", "")).strip()
        ]
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
            self._set_busy(False, "Falha")
            QMessageBox.critical(self, "Falha no processamento", message)

        worker.signals.finished.connect(finished)
        worker.signals.failed.connect(failed)
        self.thread_pool.start(worker)

    def _set_busy(self, busy: bool, message: str) -> None:
        for widget in (
            self.tts_generate_button,
            self.tts_play_button,
            self.voice_button,
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

    window.show()
    if clipboard_requested:
        QTimer.singleShot(250, window._generate_audio_from_clipboard)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
