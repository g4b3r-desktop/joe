from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QSettings, QThreadPool, QUrl, Signal
from PySide6.QtGui import QAction, QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from joe.services.colab_api import ColabAPIClient
from joe.services.parakeet import ParakeetReduxService
from joe.services.pocket_tts import PocketTTSService
from joe.utils import build_srt


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
        except Exception as exc:
            details = traceback.format_exc()
            self.signals.failed.emit(f"{exc}\n\n{details}")


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

        self.generated_audio: Path | None = None
        self.transcript_segments: list[dict] = []
        self.transcription_backend = ""

        self.player = QMediaPlayer(self)
        self.audio_output = QAudioOutput(self)
        self.player.setAudioOutput(self.audio_output)

        self._build_ui()
        self.setStyleSheet(self._stylesheet())
        self.statusBar().showMessage("Pronto — os modelos serão carregados sob demanda.")

    def _build_ui(self) -> None:
        tabs = QTabWidget()
        tabs.addTab(self._build_tts_tab(), "Texto → Áudio")
        tabs.addTab(self._build_asr_tab(), "Áudio → Texto")
        tabs.addTab(self._build_colab_tab(), "Google Colab / API")
        self.setCentralWidget(tabs)

        bar = QStatusBar()
        self.setStatusBar(bar)
        self.status_label = QLabel("Pronto")
        bar.addPermanentWidget(self.status_label)

        file_menu = self.menuBar().addMenu("Arquivo")
        exit_action = QAction("Sair", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

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

        self.colab_token = QLineEdit()
        self.colab_token.setEchoMode(QLineEdit.EchoMode.Password)
        self.colab_token.setPlaceholderText("Cole o token gerado pelo notebook")

        form = QFormLayout()
        form.addRow("Modo de execução", self.processing_mode)
        form.addRow("URL da API", self.colab_url)
        form.addRow("Token de acesso", self.colab_token)

        self.colab_test_button = QPushButton("Testar conexão com o Colab")
        self.colab_test_button.setObjectName("primaryButton")
        self.colab_test_button.clicked.connect(self._test_colab_connection)

        self.colab_open_button = QPushButton("Abrir notebook no Google Colab")
        self.colab_open_button.clicked.connect(self._open_colab_notebook)

        self.colab_info = QLabel(
            "1. Abra o notebook no Google Colab e execute as células.\\n"
            "2. Copie a URL trycloudflare.com e o token exibidos no final.\\n"
            "3. Cole os dois campos acima, teste a conexão e selecione Google Colab no modo.\\n\\n"
            "A URL e o token não são iguais: a URL muda quando o túnel é reiniciado. "
            "O token fica apenas na memória do Joe e não é gravado nas configurações."
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

    def _update_mode_widgets(self) -> None:
        if hasattr(self, "asr_device") and hasattr(self, "processing_mode"):
            self.asr_device.setEnabled(self.processing_mode.currentData() == "local")

    def _colab_client(self) -> ColabAPIClient:
        return ColabAPIClient(self.colab_url.text(), self.colab_token.text())

    def _test_colab_connection(self) -> None:
        try:
            client = self._colab_client()
        except ValueError as exc:
            QMessageBox.information(self, "Configuração incompleta", str(exc))
            return
        self._run_task(
            "Testando conexão com a API do Colab…",
            client.test_connection,
            self._colab_connection_finished,
        )

    def _colab_connection_finished(self, result: object) -> None:
        data = result if isinstance(result, dict) else {}
        gpu = data.get("gpu_name") or ("CUDA disponível" if data.get("cuda_available") else "CPU")
        self.colab_info.setText(
            f"Conexão confirmada. Servidor: {data.get('service', 'Joe Colab API')}\\n"
            f"ASR: {data.get('asr', 'Parakeet Redux')}\\n"
            f"GPU: {gpu}\\n"
            f"Parakeet carregado: {'sim' if data.get('asr_loaded') else 'ainda não'}"
        )
        self.statusBar().showMessage("API do Google Colab conectada.", 5000)

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
            "Gere um arquivo WAV localmente com o Kyutai Pocket TTS."
        )
        description.setObjectName("muted")

        self.tts_text = QTextEdit()
        self.tts_text.setPlaceholderText(
            "Digite ou cole aqui o texto que deseja transformar em áudio..."
        )
        self.tts_text.setMinimumHeight(240)

        self.tts_language = QComboBox()
        self.tts_language.addItem("Português", "portuguese")
        self.tts_language.addItem("Inglês", "english")
        self.tts_language.addItem("Espanhol", "spanish")
        self.tts_language.addItem("Francês", "french")
        self.tts_language.addItem("Italiano", "italian")
        self.tts_language.addItem("Alemão", "german")
        self.tts_language.addItem("Neerlandês", "dutch")

        self.voice_path = QLineEdit()
        self.voice_path.setReadOnly(True)
        self.voice_path.setPlaceholderText("Voz padrão do Pocket TTS")
        self.voice_button = QPushButton("Escolher áudio de referência…")
        self.voice_button.clicked.connect(self._choose_voice)

        options = QFormLayout()
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
            "A primeira geração pode baixar os pesos do modelo. "
            "Depois, eles ficam armazenados no cache local."
        )
        self.tts_info.setObjectName("muted")
        self.tts_info.setWordWrap(True)

        layout.addWidget(title)
        layout.addWidget(description)
        layout.addWidget(self.tts_text, 1)
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

        self.audio_path = QLineEdit()
        self.audio_path.setReadOnly(True)
        self.audio_path.setPlaceholderText("Nenhum arquivo selecionado")
        self.audio_choose_button = QPushButton("Selecionar áudio…")
        self.audio_choose_button.clicked.connect(self._choose_audio)

        file_row = QHBoxLayout()
        file_row.addWidget(self.audio_path, 1)
        file_row.addWidget(self.audio_choose_button)

        self.timestamps_check = QCheckBox("Incluir tempos dos segmentos (quando disponível)")
        self.timestamps_check.setChecked(True)
        self.asr_device = QComboBox()
        self.asr_device.addItem("CPU (preserva a VRAM)", "cpu")
        self.asr_device.addItem("NVIDIA GPU (CUDA)", "cuda")

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

    def _generate_audio(self) -> None:
        text = self.tts_text.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Texto vazio", "Digite o texto que deseja converter.")
            return

        output, _ = QFileDialog.getSaveFileName(
            self,
            "Salvar áudio gerado",
            str(Path.home() / "joe-audio.wav"),
            "Áudio WAV (*.wav)",
        )
        if not output:
            return
        if not output.lower().endswith(".wav"):
            output += ".wav"

        language = str(self.tts_language.currentData())
        voice = self.voice_path.text().strip() or None
        if self.processing_mode.currentData() == "colab":
            try:
                remote = self._colab_client()
            except ValueError as exc:
                QMessageBox.information(self, "Configure a API do Colab", str(exc))
                return

            def generate_remote() -> Path:
                audio_data = remote.generate_audio(text, language, voice)
                destination = Path(output)
                destination.write_bytes(audio_data)
                return destination

            task = generate_remote
            status = "Gerando áudio no Google Colab…"
        else:
            task = lambda: self.tts.generate(text, language, output, voice)
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
            self.audio_choose_button,
            self.asr_button,
            self.asr_device,
            self.timestamps_check,
            self.colab_test_button if hasattr(self, "colab_test_button") else self.tts_generate_button,
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

    def closeEvent(self, event) -> None:  # noqa: N802
        if self.thread_pool.activeThreadCount():
            self.statusBar().showMessage("Aguardando a tarefa atual terminar…")
            self.thread_pool.waitForDone(-1)
        self.player.stop()
        self.asr.close()
        self.tts.close()
        super().closeEvent(event)


def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Joe Audio Studio")
    app.setOrganizationName("g4b3r-desktop")
    window = JoeWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
