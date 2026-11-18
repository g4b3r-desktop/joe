from __future__ import annotations

from pathlib import Path
from typing import Callable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Qt, QUrl, Signal
from PySide6.QtGui import QPainter, QPen
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from joe.audio_performance import format_duration
from joe.audio_waveform import load_waveform_peaks


class WaveformWidget(QWidget):
    """Small clickable amplitude overview of the current audio file."""

    seek_requested = Signal(float)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._peaks: list[float] = []
        self._progress = 0.0
        self.setMinimumHeight(68)
        self.setMaximumHeight(92)
        self.setToolTip("Clique na forma de onda para saltar para esse ponto do áudio.")

    def set_peaks(self, peaks: list[float]) -> None:
        self._peaks = [max(0.0, min(1.0, float(value))) for value in peaks]
        self.update()

    def clear(self) -> None:
        self._peaks = []
        self._progress = 0.0
        self.update()

    def set_progress(self, ratio: float) -> None:
        self._progress = max(0.0, min(1.0, float(ratio)))
        self.update()

    def paintEvent(self, _event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        bounds = self.rect().adjusted(5, 5, -5, -5)
        center_y = bounds.center().y()
        inactive = self.palette().color(self.palette().ColorRole.Mid)
        active = self.palette().color(self.palette().ColorRole.Highlight)
        width = max(1, bounds.width())

        if not self._peaks:
            painter.setPen(QPen(inactive, 2))
            painter.drawLine(bounds.left(), center_y, bounds.right(), center_y)
            painter.end()
            return

        step = max(1, width // len(self._peaks))
        for index, peak in enumerate(self._peaks):
            x = bounds.left() + int(index * width / len(self._peaks))
            height = max(2, int(peak * bounds.height() * 0.92))
            color = active if (index / len(self._peaks)) <= self._progress else inactive
            painter.setPen(QPen(color, max(1, step - 1), Qt.PenStyle.SolidLine))
            painter.drawLine(x, center_y - height // 2, x, center_y + height // 2)
        painter.end()

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.width() > 0:
            ratio = (event.position().x() - 5) / max(1, self.width() - 10)
            self.seek_requested.emit(max(0.0, min(1.0, ratio)))
            event.accept()
            return
        super().mousePressEvent(event)


class WaveformWorkerSignals(QObject):
    finished = Signal(str, object)
    failed = Signal(str, str)


class WaveformWorker(QRunnable):
    """Builds a waveform off the GUI thread using bounded memory."""

    def __init__(self, audio_path: str) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.signals = WaveformWorkerSignals()

    def run(self) -> None:
        try:
            peaks = load_waveform_peaks(self.audio_path)
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(self.audio_path, str(exc))
            return
        self.signals.finished.emit(self.audio_path, peaks)


class AudioPlayerWidget(QWidget):
    """Reusable controls for the shared Joe QMediaPlayer instance."""

    def __init__(
        self,
        player: QMediaPlayer,
        audio_output: QAudioOutput,
        *,
        volume: int = 80,
        playback_rate: float = 1.0,
        on_volume_changed: Callable[[int], None] | None = None,
        on_rate_changed: Callable[[float], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.player = player
        self.audio_output = audio_output
        self.on_volume_changed = on_volume_changed
        self.on_rate_changed = on_rate_changed
        self._seeking = False
        self._waveform_path = ""
        self._waveform_pool = QThreadPool(self)
        self._waveform_pool.setMaxThreadCount(1)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 6, 0, 0)
        layout.setSpacing(6)

        self.waveform = WaveformWidget()
        self.waveform.seek_requested.connect(self._seek_ratio)
        layout.addWidget(self.waveform)

        timeline = QHBoxLayout()
        self.position_label = QLabel("0,0s")
        self.duration_label = QLabel("0,0s")
        self.seek_slider = QSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setSingleStep(1000)
        self.seek_slider.sliderPressed.connect(self._begin_seek)
        self.seek_slider.sliderReleased.connect(self._finish_seek)
        self.seek_slider.sliderMoved.connect(self._preview_seek)
        timeline.addWidget(self.position_label)
        timeline.addWidget(self.seek_slider, 1)
        timeline.addWidget(self.duration_label)
        layout.addLayout(timeline)

        controls = QHBoxLayout()
        self.play_button = QPushButton("▶ Reproduzir")
        self.play_button.clicked.connect(self.toggle_playback)
        self.stop_button = QPushButton("■ Parar")
        self.stop_button.clicked.connect(self.stop_playback)

        self.speed_combo = QComboBox()
        for label, rate in (
            ("0,75×", 0.75),
            ("1×", 1.0),
            ("1,25×", 1.25),
            ("1,5×", 1.5),
            ("2×", 2.0),
        ):
            self.speed_combo.addItem(label, rate)
        rate_index = self.speed_combo.findData(float(playback_rate))
        self.speed_combo.setCurrentIndex(max(0, rate_index))
        self.speed_combo.currentIndexChanged.connect(self._change_rate)

        self.volume_label = QLabel("Volume")
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setFixedWidth(108)
        self.volume_slider.setValue(max(0, min(100, int(volume))))
        self.volume_slider.valueChanged.connect(self._change_volume)
        controls.addWidget(self.play_button)
        controls.addWidget(self.stop_button)
        controls.addStretch(1)
        controls.addWidget(QLabel("Velocidade"))
        controls.addWidget(self.speed_combo)
        controls.addWidget(self.volume_label)
        controls.addWidget(self.volume_slider)
        layout.addLayout(controls)

        self.player.positionChanged.connect(self._update_position)
        self.player.durationChanged.connect(self._update_duration)
        self.player.playbackStateChanged.connect(self._update_play_button)
        self.player.sourceChanged.connect(self._source_changed)

        self.audio_output.setVolume(self.volume_slider.value() / 100)
        self.player.setPlaybackRate(float(self.speed_combo.currentData()))
        self._update_play_button(self.player.playbackState())

    def set_audio_path(self, audio_path: str | Path) -> None:
        path = str(Path(audio_path).expanduser().resolve())
        if path == self._waveform_path:
            return
        self._waveform_path = path
        self.waveform.clear()
        worker = WaveformWorker(path)
        worker.signals.finished.connect(self._waveform_ready)
        worker.signals.failed.connect(self._waveform_failed)
        self._waveform_pool.start(worker)

    def play_file(self, audio_path: str | Path) -> None:
        path = Path(audio_path).expanduser().resolve()
        if not path.is_file():
            return
        source = QUrl.fromLocalFile(str(path))
        same_source = self.player.source() == source
        if same_source and self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
            return
        if not same_source:
            self.set_audio_path(path)
            self.player.setSource(source)
        elif self._waveform_path != str(path):
            self.set_audio_path(path)
        if (
            self.player.duration() > 0
            and self.player.position() >= self.player.duration()
        ):
            self.player.setPosition(0)
        self.player.play()

    def toggle_playback(self) -> None:
        if not self.player.source().isValid():
            return
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            if self.player.duration() > 0 and self.player.position() >= self.player.duration():
                self.player.setPosition(0)
            self.player.play()

    def stop_playback(self) -> None:
        self.player.stop()
        self.player.setPosition(0)

    def _source_changed(self, source: QUrl) -> None:
        source_path = source.toLocalFile() if source.isLocalFile() else ""
        normalized = str(Path(source_path).resolve()) if source_path else ""
        if normalized != self._waveform_path:
            self._waveform_path = ""
            self.waveform.clear()

    def _waveform_ready(self, path: str, peaks: object) -> None:
        if path == self._waveform_path and isinstance(peaks, list):
            self.waveform.set_peaks(peaks)

    def _waveform_failed(self, path: str, message: str) -> None:
        if path == self._waveform_path:
            self.waveform.setToolTip(f"Não foi possível gerar a forma de onda: {message}")

    def _begin_seek(self) -> None:
        self._seeking = True

    def _preview_seek(self, position: int) -> None:
        self.position_label.setText(format_duration(position / 1000))

    def _finish_seek(self) -> None:
        self._seeking = False
        self.player.setPosition(self.seek_slider.value())

    def _seek_ratio(self, ratio: float) -> None:
        duration = self.player.duration()
        if duration > 0:
            self.player.setPosition(int(duration * ratio))

    def _update_position(self, position: int) -> None:
        if not self._seeking:
            self.seek_slider.setValue(position)
            self.position_label.setText(format_duration(position / 1000))
        duration = self.player.duration()
        self.waveform.set_progress(position / duration if duration > 0 else 0.0)

    def _update_duration(self, duration: int) -> None:
        self.seek_slider.setRange(0, max(0, duration))
        self.duration_label.setText(format_duration(duration / 1000))

    def _update_play_button(self, state) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setText("⏸ Pausar" if playing else "▶ Reproduzir")

    def _change_volume(self, value: int) -> None:
        self.audio_output.setVolume(value / 100)
        if self.on_volume_changed is not None:
            self.on_volume_changed(value)

    def _change_rate(self, _index: int) -> None:
        rate = float(self.speed_combo.currentData())
        self.player.setPlaybackRate(rate)
        if self.on_rate_changed is not None:
            self.on_rate_changed(rate)
