from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, QUrl, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel,
    QPushButton, QSlider, QVBoxLayout, QWidget,
)

from joe.audio_editing import (
    export_trimmed_audio, export_without_silence, find_silence_regions,
    load_audio, waveform_peaks,
)


class _TaskSignals(QObject):
    finished = Signal(object)
    failed = Signal(str)


class _FunctionTask(QRunnable):
    def __init__(self, function: Callable[[], object]) -> None:
        super().__init__()
        self.function = function
        self.signals = _TaskSignals()

    def run(self) -> None:
        try:
            self.signals.finished.emit(self.function())
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(str(exc))


class WaveformWidget(QWidget):
    seek_requested = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumHeight(76)
        self.setMaximumHeight(112)
        self.setToolTip("Clique na forma de onda para navegar pelo áudio.")
        self.peaks: list[float] = []
        self.duration_ms = 0
        self.position_ms = 0
        self.start_marker_ms: int | None = None
        self.end_marker_ms: int | None = None

    def set_audio(self, peaks: list[float], duration_ms: int) -> None:
        self.peaks = peaks
        self.duration_ms = max(0, duration_ms)
        self.update()

    def set_position(self, position_ms: int) -> None:
        self.position_ms = max(0, position_ms)
        self.update()

    def set_markers(self, start_ms: int | None, end_ms: int | None) -> None:
        self.start_marker_ms, self.end_marker_ms = start_ms, end_ms
        self.update()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if (event.button() == Qt.MouseButton.LeftButton
                and self.duration_ms > 0 and self.width() > 16):
            fraction = min(1.0, max(0.0,
                (event.position().x() - 8) / (self.width() - 16)))
            self.seek_requested.emit(int(fraction * self.duration_ms))
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#171b24"))
        left, width = 8, max(1, self.width() - 16)
        center = self.height() // 2
        amplitude = max(8, self.height() // 2 - 7)
        if not self.peaks:
            painter.setPen(QColor("#80899a"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                "A forma de onda será exibida para arquivos compatíveis")
            return
        progress_x = (left + int(width * self.position_ms / self.duration_ms)
                      if self.duration_ms > 0 else left)
        denominator = max(1, len(self.peaks) - 1)
        for index, peak in enumerate(self.peaks):
            x = left + int(index * width / denominator)
            height = max(1, int(min(1.0, peak) * amplitude))
            painter.setPen(QPen(
                QColor("#a8b7ff") if x <= progress_x else QColor("#586477"), 1
            ))
            painter.drawLine(x, center - height, x, center + height)
        for marker, color in (
            (self.start_marker_ms, QColor("#4fd19a")),
            (self.end_marker_ms, QColor("#ffb45f")),
        ):
            if marker is not None and self.duration_ms > 0:
                x = left + int(width * marker / self.duration_ms)
                painter.setPen(QPen(color, 2))
                painter.drawLine(x, 3, x, self.height() - 3)


class AdvancedAudioPlayer(QWidget):
    """Advanced controls attached to Joe's shared QMediaPlayer and QAudioOutput."""

    status_message = Signal(str)

    def __init__(self, player: QMediaPlayer, audio_output: QAudioOutput,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.player, self.audio_output = player, audio_output
        self.current_path: Path | None = None
        self.duration_ms = 0
        self.start_marker_ms: int | None = None
        self.end_marker_ms: int | None = None
        self._loading_generation = 0
        self._workers: set[_FunctionTask] = set()
        self._busy = False
        self._pool = QThreadPool(self)
        self._pool.setMaxThreadCount(1)
        self._build_ui()
        self.player.sourceChanged.connect(self._source_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.positionChanged.connect(self._position_changed)
        self.player.playbackStateChanged.connect(self._playback_state_changed)
        self.audio_output.volumeChanged.connect(self._volume_changed)
        self._source_changed(self.player.source())
        self._volume_changed(self.audio_output.volume())
        self._playback_state_changed(self.player.playbackState())

    @staticmethod
    def _format_time(milliseconds: int) -> str:
        total = max(0, int(milliseconds // 1000))
        minutes, seconds = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(12)
        title = QLabel("Player avançado")
        title.setObjectName("pageTitle")
        note = QLabel(
            "O reprodutor é compartilhado com a geração, a biblioteca de vozes e o histórico. "
            "Cortes e redução de pausas geram uma cópia; o original não é sobrescrito."
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(note)

        current = QHBoxLayout()
        self.file_label = QLabel("Nenhum áudio selecionado")
        self.file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        current.addWidget(self.file_label, 1)
        current.addWidget(self.time_label)
        layout.addLayout(current)

        self.waveform = WaveformWidget()
        self.waveform.seek_requested.connect(self._seek)
        layout.addWidget(self.waveform)

        seek = QHBoxLayout()
        self.play_button = QPushButton("Reproduzir")
        self.play_button.setObjectName("primaryButton")
        self.play_button.clicked.connect(self._toggle_playback)
        self.stop_button = QPushButton("Parar")
        self.stop_button.clicked.connect(self.player.stop)
        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 1000)
        self.position_slider.setToolTip("Posição da reprodução")
        self.position_slider.sliderMoved.connect(self._seek_slider)
        seek.addWidget(self.play_button)
        seek.addWidget(self.stop_button)
        seek.addWidget(self.position_slider, 1)
        layout.addLayout(seek)

        transport = QHBoxLayout()
        for label, delta in (("−10 s", -10), ("−5 s", -5), ("+5 s", 5), ("+10 s", 10)):
            button = QPushButton(label)
            button.clicked.connect(lambda _checked=False, value=delta: self._jump(value))
            transport.addWidget(button)
        transport.addSpacing(12)
        transport.addWidget(QLabel("Velocidade"))
        self.speed_combo = QComboBox()
        for rate in (0.75, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0):
            self.speed_combo.addItem(f"{rate:g}×", rate)
        self.speed_combo.setCurrentIndex(self.speed_combo.findData(1.0))
        self.speed_combo.currentIndexChanged.connect(
            lambda: self.player.setPlaybackRate(float(self.speed_combo.currentData()))
        )
        transport.addWidget(self.speed_combo)
        transport.addSpacing(12)
        transport.addWidget(QLabel("Volume"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setFixedWidth(120)
        self.volume_slider.valueChanged.connect(
            lambda value: self.audio_output.setVolume(value / 100.0))
        transport.addWidget(self.volume_slider)
        self.volume_label = QLabel("100%")
        self.volume_label.setMinimumWidth(42)
        transport.addWidget(self.volume_label)
        layout.addLayout(transport)

        marks = QHBoxLayout()
        self.mark_start_button = QPushButton("Definir início (A)")
        self.mark_end_button = QPushButton("Definir fim (B)")
        self.repeat_button = QPushButton("Repetir A–B")
        self.repeat_button.setCheckable(True)
        self.reset_marks_button = QPushButton("Limpar marcações")
        self.export_trim_button = QPushButton("Exportar corte…")
        self.mark_start_button.clicked.connect(self._mark_start)
        self.mark_end_button.clicked.connect(self._mark_end)
        self.repeat_button.toggled.connect(self._toggle_repeat)
        self.reset_marks_button.clicked.connect(self._reset_markers)
        self.export_trim_button.clicked.connect(self._export_trim)
        for button in (self.mark_start_button, self.mark_end_button,
                       self.repeat_button, self.reset_marks_button,
                       self.export_trim_button):
            marks.addWidget(button)
        layout.addLayout(marks)

        silence = QHBoxLayout()
        silence.addWidget(QLabel("Silêncio abaixo de"))
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(-80, -10)
        self.threshold_spin.setDecimals(0)
        self.threshold_spin.setValue(-40)
        self.threshold_spin.setSuffix(" dB")
        silence.addWidget(self.threshold_spin)
        silence.addWidget(QLabel("a partir de"))
        self.minimum_silence_spin = QDoubleSpinBox()
        self.minimum_silence_spin.setRange(0.1, 30)
        self.minimum_silence_spin.setDecimals(1)
        self.minimum_silence_spin.setSingleStep(0.1)
        self.minimum_silence_spin.setValue(0.5)
        self.minimum_silence_spin.setSuffix(" s")
        silence.addWidget(self.minimum_silence_spin)
        silence.addWidget(QLabel("Manter"))
        self.keep_silence_spin = QDoubleSpinBox()
        self.keep_silence_spin.setRange(0, 1)
        self.keep_silence_spin.setDecimals(2)
        self.keep_silence_spin.setSingleStep(0.05)
        self.keep_silence_spin.setValue(0.15)
        self.keep_silence_spin.setSuffix(" s")
        silence.addWidget(self.keep_silence_spin)
        self.detect_silence_button = QPushButton("Detectar silêncios")
        self.remove_silence_button = QPushButton("Remover silêncios e salvar…")
        self.detect_silence_button.clicked.connect(self._detect_silences)
        self.remove_silence_button.clicked.connect(self._remove_silences)
        silence.addWidget(self.detect_silence_button)
        silence.addWidget(self.remove_silence_button)
        layout.addLayout(silence)

        self.status_label = QLabel("Selecione ou reproduza um áudio para começar.")
        self.status_label.setObjectName("muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self._update_action_state()

    def _run_task(self, function: Callable[[], object],
                  on_success: Callable[[object], None],
                  on_failure: Callable[[str], None] | None = None) -> None:
        worker = _FunctionTask(function)
        self._workers.add(worker)
        self._busy = True
        self._update_action_state()
        self.status_label.setText("Processando áudio…")

        def completed(result: object, task: _FunctionTask = worker) -> None:
            self._workers.discard(task)
            self._busy = bool(self._workers)
            self._update_action_state()
            on_success(result)

        def failed(message: str, task: _FunctionTask = worker) -> None:
            self._workers.discard(task)
            self._busy = bool(self._workers)
            self._update_action_state()
            if on_failure:
                on_failure(message)
            else:
                self.status_label.setText(f"Falha na operação: {message}")
                self.status_message.emit(f"Falha no player: {message}")

        worker.signals.finished.connect(completed, Qt.ConnectionType.QueuedConnection)
        worker.signals.failed.connect(failed, Qt.ConnectionType.QueuedConnection)
        self._pool.start(worker)

    def _source_changed(self, source: QUrl) -> None:
        self._loading_generation += 1
        generation = self._loading_generation
        self.current_path = Path(source.toLocalFile()) if source.isLocalFile() else None
        self.duration_ms = 0
        self.start_marker_ms = self.end_marker_ms = None
        self.repeat_button.setChecked(False)
        self.position_slider.setValue(0)
        self.waveform.set_audio([], 0)
        self.waveform.set_markers(None, None)
        self._update_time_label()

        if self.current_path is None:
            self.file_label.setText("Nenhum áudio local selecionado")
            self._update_action_state()
            return
        path = self.current_path
        self.file_label.setText(path.name)
        if not path.is_file():
            self.status_label.setText("O arquivo selecionado não está mais disponível.")
            self._update_action_state()
            return

        def load() -> tuple[int, list[float], int]:
            samples, sample_rate = load_audio(path)
            duration = int(round(samples.shape[0] * 1000 / sample_rate))
            return generation, waveform_peaks(samples), duration

        def loaded(result: object) -> None:
            token, peaks, duration = result
            if token != self._loading_generation:
                return
            self.duration_ms = int(duration)
            self.waveform.set_audio(list(peaks), self.duration_ms)
            self.status_label.setText("Waveform pronta. Clique nela para navegar.")
            self._update_action_state()

        def load_failed(_message: str) -> None:
            if generation == self._loading_generation:
                self.status_label.setText(
                    "Waveform indisponível para este formato; a reprodução continua "
                    "usando o decodificador do sistema."
                )
                self._update_action_state()

        self._run_task(load, loaded, load_failed)

    def _duration_changed(self, duration: int) -> None:
        self.duration_ms = max(0, int(duration))
        self.waveform.duration_ms = self.duration_ms
        self.waveform.update()
        self._update_time_label()
        self._update_action_state()

    def _position_changed(self, position: int) -> None:
        position = max(0, int(position))
        self.waveform.set_position(position)
        if self.duration_ms > 0:
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(min(1000, int(position * 1000 / self.duration_ms)))
            self.position_slider.blockSignals(False)
        self._update_time_label()
        if (self.repeat_button.isChecked() and self.start_marker_ms is not None
                and self.end_marker_ms is not None and position >= self.end_marker_ms):
            self.player.setPosition(self.start_marker_ms)

    def _playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        self.play_button.setText(
            "Pausar" if state == QMediaPlayer.PlaybackState.PlayingState
            else "Continuar" if state == QMediaPlayer.PlaybackState.PausedState
            else "Reproduzir"
        )

    def _volume_changed(self, volume: float) -> None:
        value = min(100, max(0, int(round(volume * 100))))
        self.volume_slider.blockSignals(True)
        self.volume_slider.setValue(value)
        self.volume_slider.blockSignals(False)
        self.volume_label.setText(f"{value}%")

    def _update_time_label(self) -> None:
        self.time_label.setText(
            f"{self._format_time(self.player.position())} / "
            f"{self._format_time(self.duration_ms)}"
        )

    def _seek(self, position: int) -> None:
        if self.duration_ms > 0:
            self.player.setPosition(min(self.duration_ms, max(0, position)))

    def _seek_slider(self, value: int) -> None:
        if self.duration_ms > 0:
            self._seek(int(self.duration_ms * value / 1000))

    def _toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            if self.duration_ms > 0 and self.player.position() >= self.duration_ms:
                self.player.setPosition(0)
            self.player.play()

    def _jump(self, seconds: int) -> None:
        self._seek(self.player.position() + seconds * 1000)

    def _mark_start(self) -> None:
        position = max(0, self.player.position())
        if self.end_marker_ms is not None and position >= self.end_marker_ms:
            self.end_marker_ms = None
            self.repeat_button.setChecked(False)
        self.start_marker_ms = position
        self._update_markers()
        self._update_action_state()
        self.status_label.setText(f"Início A definido em {self._format_time(position)}.")

    def _mark_end(self) -> None:
        position = max(0, self.player.position())
        if self.start_marker_ms is not None and position <= self.start_marker_ms:
            self.status_label.setText("O fim B precisa ficar depois do início A.")
            return
        self.end_marker_ms = position
        self._update_markers()
        self._update_action_state()
        self.status_label.setText(f"Fim B definido em {self._format_time(position)}.")

    def _toggle_repeat(self, checked: bool) -> None:
        if checked and (self.start_marker_ms is None or self.end_marker_ms is None
                        or self.end_marker_ms <= self.start_marker_ms):
            self.repeat_button.blockSignals(True)
            self.repeat_button.setChecked(False)
            self.repeat_button.blockSignals(False)
            self.status_label.setText("Defina A e B em posições válidas antes de repetir.")
            return
        if checked:
            self._seek(self.start_marker_ms or 0)
            self.player.play()

    def _reset_markers(self) -> None:
        self.start_marker_ms = self.end_marker_ms = None
        self.repeat_button.setChecked(False)
        self._update_markers()
        self._update_action_state()
        self.status_label.setText("Marcações A e B removidas.")

    def _update_markers(self) -> None:
        self.waveform.set_markers(self.start_marker_ms, self.end_marker_ms)

    def _export_trim(self) -> None:
        path = self.current_path
        if path is None or not path.is_file():
            return
        start = self.start_marker_ms if self.start_marker_ms is not None else 0
        end = self.end_marker_ms if self.end_marker_ms is not None else self.duration_ms
        if end <= start:
            self.status_label.setText("Defina um intervalo A–B válido para exportar o corte.")
            return
        output, _ = QFileDialog.getSaveFileName(
            self, "Exportar trecho cortado",
            str(path.with_name(f"{path.stem}_cortado.wav")), "Áudio WAV (*.wav)"
        )
        if not output:
            return
        destination = Path(output)
        if destination.suffix.lower() != ".wav":
            destination = destination.with_suffix(".wav")
        self._run_task(
            lambda: export_trimmed_audio(path, destination, start / 1000, end / 1000),
            lambda saved: self._operation_finished(f"Corte exportado em: {saved}"),
        )

    def _silence_options(self) -> dict[str, float]:
        return {
            "threshold_db": self.threshold_spin.value(),
            "minimum_silence_seconds": self.minimum_silence_spin.value(),
            "keep_silence_seconds": self.keep_silence_spin.value(),
        }

    def _detect_silences(self) -> None:
        path = self.current_path
        if path is None or not path.is_file():
            return
        options = self._silence_options()

        def detect() -> tuple[int, float]:
            samples, rate = load_audio(path)
            regions = find_silence_regions(
                samples, rate, threshold_db=options["threshold_db"],
                minimum_silence_seconds=options["minimum_silence_seconds"],
            )
            return len(regions), sum(end - start for start, end in regions) / rate

        def detected(result: object) -> None:
            if self.current_path != path:
                return
            count, seconds = result
            self.status_label.setText(
                f"Silêncios: {count} trecho(s), {seconds:.2f} s no total. "
                "O original não foi alterado."
            )

        self._run_task(detect, detected)

    def _remove_silences(self) -> None:
        path = self.current_path
        if path is None or not path.is_file():
            return
        output, _ = QFileDialog.getSaveFileName(
            self, "Salvar áudio sem silêncios excessivos",
            str(path.with_name(f"{path.stem}_sem_silencios.wav")), "Áudio WAV (*.wav)"
        )
        if not output:
            return
        destination = Path(output)
        if destination.suffix.lower() != ".wav":
            destination = destination.with_suffix(".wav")
        options = self._silence_options()

        def process():
            return export_without_silence(path, destination, **options)

        def processed(result: object) -> None:
            saved, removed, count, before, after = result
            self._operation_finished(
                f"Redução concluída: {count} pausa(s), {removed:.2f} s removidos; "
                f"{before:.2f} s → {after:.2f} s. Cópia: {saved}"
            )

        self._run_task(process, processed)

    def _operation_finished(self, message: str) -> None:
        self.status_label.setText(message)
        self.status_message.emit(message)

    def _update_action_state(self) -> None:
        has_source = not self.player.source().isEmpty()
        available = bool(self.current_path is not None and self.current_path.is_file())
        self.play_button.setEnabled(has_source)
        self.stop_button.setEnabled(has_source)
        self.position_slider.setEnabled(self.duration_ms > 0)
        self.speed_combo.setEnabled(has_source)
        self.mark_start_button.setEnabled(available and not self._busy)
        self.mark_end_button.setEnabled(available and not self._busy)
        self.repeat_button.setEnabled(
            not self._busy and self.start_marker_ms is not None
            and self.end_marker_ms is not None
        )
        self.reset_marks_button.setEnabled(
            not self._busy and (self.start_marker_ms is not None or self.end_marker_ms is not None)
        )
        self.export_trim_button.setEnabled(
            available and not self._busy and self.duration_ms > 0
        )
        self.detect_silence_button.setEnabled(available and not self._busy)
        self.remove_silence_button.setEnabled(available and not self._busy)
        if self._busy:
            self.status_label.setText("Processando áudio…")
