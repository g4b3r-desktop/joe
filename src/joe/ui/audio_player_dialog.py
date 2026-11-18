from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
)

from joe.audio_performance import format_duration


class AudioPlayerDialog(QDialog):
    """Advanced controls for the shared application-wide audio player."""

    def __init__(self, player, audio_output, parent=None) -> None:
        super().__init__(parent)
        self.player = player
        self.audio_output = audio_output
        self._updating_position = False
        self.setWindowTitle("Player de áudio — Joe")
        self.resize(620, 250)

        layout = QVBoxLayout(self)
        self.source_label = QLabel("Nenhum áudio selecionado")
        self.source_label.setWordWrap(True)
        layout.addWidget(self.source_label)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, 0)
        self.position_slider.sliderMoved.connect(self.player.setPosition)
        layout.addWidget(self.position_slider)

        time_row = QHBoxLayout()
        self.position_label = QLabel("0,0s")
        self.duration_label = QLabel("0,0s")
        time_row.addWidget(self.position_label)
        time_row.addStretch(1)
        time_row.addWidget(self.duration_label)
        layout.addLayout(time_row)

        controls = QHBoxLayout()
        self.play_button = QPushButton("▶ Reproduzir")
        self.play_button.clicked.connect(self.player.play)
        self.pause_button = QPushButton("Pausar")
        self.pause_button.clicked.connect(self.player.pause)
        self.stop_button = QPushButton("Parar")
        self.stop_button.clicked.connect(self.player.stop)
        self.back_button = QPushButton("−5s")
        self.back_button.clicked.connect(lambda: self._skip(-5000))
        self.forward_button = QPushButton("+5s")
        self.forward_button.clicked.connect(lambda: self._skip(5000))
        for button in (
            self.back_button, self.play_button, self.pause_button,
            self.stop_button, self.forward_button
        ):
            controls.addWidget(button)
        layout.addLayout(controls)

        settings = QHBoxLayout()
        settings.addWidget(QLabel("Velocidade"))
        self.speed_combo = QComboBox()
        for value in (0.75, 1.0, 1.25, 1.5, 2.0):
            self.speed_combo.addItem(f"{value:g}×", value)
        self.speed_combo.setCurrentIndex(self.speed_combo.findData(1.0))
        self.speed_combo.currentIndexChanged.connect(
            lambda: self.player.setPlaybackRate(float(self.speed_combo.currentData()))
        )
        settings.addWidget(self.speed_combo)

        settings.addWidget(QLabel("Volume"))
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(round(self.audio_output.volume() * 100))
        self.volume_slider.valueChanged.connect(
            lambda value: self.audio_output.setVolume(value / 100.0)
        )
        settings.addWidget(self.volume_slider, 1)
        layout.addLayout(settings)

        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.sourceChanged.connect(self._source_changed)
        self._source_changed(self.player.source())
        self._duration_changed(self.player.duration())
        self._position_changed(self.player.position())

    def _skip(self, milliseconds: int) -> None:
        duration = self.player.duration()
        target = max(0, self.player.position() + milliseconds)
        if duration > 0:
            target = min(target, duration)
        self.player.setPosition(target)

    def _position_changed(self, position: int) -> None:
        if not self.position_slider.isSliderDown():
            self._updating_position = True
            self.position_slider.setValue(position)
            self._updating_position = False
        self.position_label.setText(format_duration(position / 1000.0))

    def _duration_changed(self, duration: int) -> None:
        self.position_slider.setRange(0, max(0, duration))
        self.duration_label.setText(format_duration(duration / 1000.0))

    def _source_changed(self, source: QUrl) -> None:
        path = source.toLocalFile()
        if path:
            self.source_label.setText(Path(path).name)
            self.source_label.setToolTip(path)
        else:
            self.source_label.setText("Nenhum áudio selecionado")
            self.source_label.setToolTip("")
        self._position_changed(self.player.position())
        self._duration_changed(self.player.duration())

