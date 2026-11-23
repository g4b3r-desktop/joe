from __future__ import annotations

import shutil
from pathlib import Path

from PySide6.QtCore import QSettings, QSize, QUrl, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QShortcut, QTextDocument, QTextOption
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QMessageBox,
    QPushButton,
    QSlider,
    QStyle,
    QStyledItemDelegate,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from joe.audio_performance import AudioPerformance, format_duration, wav_duration_seconds
from joe.transcript_sync import normalize_timed_segments


# Paleta única da janela do atalho. O delegate e o QSS reutilizam estas cores.
CLIPBOARD_DIALOG_PALETTE = {
    "window": "#0c1420",
    "surface": "#111d2b",
    "surface_hover": "#182a3e",
    "surface_active": "#1b3550",
    "blue": "#368de0",
    "on_blue": "#ffffff",
    "blue_soft": "#254a6d",
    "text": "#edf3fb",
    "text_secondary": "#c0cede",
    "text_muted": "#91a4ba",
    "separator": "#29394b",
    "pending": "#91a4ba",
    "success": "#9ed2b0",
    "danger": "#e59b9b",
    "slider_track": "#34485e",
    "control": "#1d3045",
    "control_hover": "#29435e",
}


def _dialog_stylesheet() -> str:
    p = CLIPBOARD_DIALOG_PALETTE
    return f"""
        QDialog {{
            background: {p['window']};
            color: {p['text']};
            font-family: "Noto Sans", "DejaVu Sans", sans-serif;
            font-size: 14px;
        }}
        QLabel {{
            background: transparent;
            color: {p['text']};
            font-size: 14px;
        }}
        QLabel#brand {{
            background: {p['blue']};
            color: {p['on_blue']};
            border-radius: 6px;
            padding: 4px 9px;
            font-size: 13px;
            font-weight: 800;
        }}
        QLabel#dialogTitle {{
            color: {p['text']};
            font-size: 16px;
            font-weight: 700;
        }}
        QLabel#preview {{
            color: {p['text_secondary']};
            font-size: 13px;
        }}
        QLabel#summary {{
            color: {p['text_secondary']};
            font-size: 13px;
        }}
        QLabel#status {{
            color: {p['text_secondary']};
            font-size: 13px;
        }}
        QLabel#detailValue {{
            color: {p['text']};
            font-size: 13px;
        }}
        QLabel#detailCaption {{
            color: {p['text_muted']};
            font-size: 12px;
        }}
        QToolButton {{
            background: transparent;
            border: 0;
            color: {p['text_secondary']};
            padding: 4px 6px;
            font-size: 13px;
        }}
        QToolButton:hover {{
            color: {p['text']};
            background: {p['surface_hover']};
        }}
        QListWidget {{
            background: transparent;
            border: 0;
            outline: none;
            padding: 0;
            color: {p['text']};
            font-size: 14px;
        }}
        QListWidget::item {{
            background: transparent;
            border: 0;
            padding: 0;
            margin: 0;
        }}
        QScrollBar:vertical {{
            background: transparent;
            width: 7px;
            margin: 2px 1px 2px 1px;
        }}
        QScrollBar::handle:vertical {{
            background: {p['slider_track']};
            min-height: 28px;
            border-radius: 3px;
        }}
        QScrollBar::handle:vertical:hover {{
            background: {p['text_muted']};
        }}
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
            height: 0;
        }}
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
            background: transparent;
        }}
        QPushButton#playButton {{
            min-width: 38px;
            min-height: 34px;
            background: {p['blue_soft']};
            border: 0;
            border-radius: 8px;
            color: {p['on_blue']};
            font-size: 17px;
            font-weight: 700;
        }}
        QPushButton#playButton:hover {{
            background: {p['blue']};
        }}
        QPushButton#playButton:disabled, QPushButton#saveButton:disabled {{
            color: {p['text_muted']};
            background: {p['surface']};
        }}
        QPushButton#saveButton {{
            min-width: 36px;
            min-height: 34px;
            background: {p['control']};
            border: 0;
            border-radius: 8px;
            color: {p['text_secondary']};
            font-size: 17px;
        }}
        QPushButton#saveButton:hover {{
            background: {p['control_hover']};
            color: {p['text']};
        }}
        QSlider::groove:horizontal {{
            height: 4px;
            background: {p['slider_track']};
            border-radius: 2px;
        }}
        QSlider::sub-page:horizontal {{
            background: {p['blue']};
            border-radius: 2px;
        }}
        QSlider::add-page:horizontal {{
            background: {p['slider_track']};
            border-radius: 2px;
        }}
        QSlider::handle:horizontal {{
            background: {p['text']};
            width: 11px;
            margin: -4px 0;
            border-radius: 5px;
        }}
        QComboBox {{
            background: {p['control']};
            border: 0;
            border-radius: 7px;
            color: {p['text_secondary']};
            padding: 6px 8px;
            min-width: 66px;
            font-size: 13px;
        }}
        QComboBox:hover {{
            background: {p['control_hover']};
        }}
        QComboBox QAbstractItemView {{
            background: {p['surface']};
            color: {p['text']};
            selection-background-color: {p['blue_soft']};
            border: 1px solid {p['separator']};
        }}
        QMenu {{
            background: {p['surface']};
            color: {p['text']};
            border: 1px solid {p['separator']};
            padding: 4px;
        }}
        QMenu::item {{
            padding: 7px 22px;
            border-radius: 4px;
        }}
        QMenu::item:selected {{
            background: {p['surface_hover']};
        }}
    """


class ClickableSeekSlider(QSlider):
    """Slider whose track jumps directly to the clicked playback position."""

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if event.button() == Qt.MouseButton.LeftButton and self.maximum() > self.minimum():
            ratio = min(1.0, max(0.0, float(event.position().x()) / max(1, self.width())))
            value = self.minimum() + round(ratio * (self.maximum() - self.minimum()))
            self.setValue(value)
            self.sliderMoved.emit(value)
            self.sliderReleased.emit()
            event.accept()
            return
        super().mousePressEvent(event)


class PhraseItemDelegate(QStyledItemDelegate):
    """Draw simple wrapped phrase rows with progress, hover and active states."""

    def _document(self, text: str, font, width: int) -> QTextDocument:
        document = QTextDocument()
        document.setDocumentMargin(0)
        document.setDefaultFont(font)
        document.setDefaultStyleSheet(
            f"body {{ color: {CLIPBOARD_DIALOG_PALETTE['text']}; }}"
        )
        document.setPlainText(text)
        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        document.setDefaultTextOption(option)
        document.setTextWidth(max(100, width))
        return document

    def sizeHint(self, option, index) -> QSize:  # noqa: N802
        view = self.parent()
        width = option.rect.width()
        if width <= 0 and isinstance(view, QListWidget):
            width = view.viewport().width()
        if width <= 0:
            width = 620
        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        document = self._document(text, option.font, width - 132)
        return QSize(width, max(48, int(document.size().height()) + 22))

    def paint(self, painter: QPainter, option, index) -> None:
        p = CLIPBOARD_DIALOG_PALETTE
        rect = option.rect
        active = bool(index.data(Qt.ItemDataRole.UserRole + 2))
        generated = bool(index.data(Qt.ItemDataRole.UserRole + 1))
        failed = bool(index.data(Qt.ItemDataRole.UserRole + 3))
        hovered = bool(option.state & QStyle.StateFlag.State_MouseOver)

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if active:
            painter.fillRect(rect, QColor(p['surface_active']))
            painter.fillRect(rect.left(), rect.top() + 4, 3, max(1, rect.height() - 8), QColor(p['blue']))
        elif hovered:
            painter.fillRect(rect, QColor(p['surface_hover']))

        separator_y = rect.bottom()
        painter.fillRect(rect.left() + 12, separator_y, max(0, rect.width() - 24), 1, QColor(p['separator']))

        number = str(index.data(Qt.ItemDataRole.UserRole) or "").zfill(3)
        painter.setPen(QColor(p['text_muted']))
        painter.drawText(
            rect.adjusted(15, 0, -rect.width() + 55, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            number,
        )

        if generated:
            marker, marker_color = "✓", p['success']
        elif failed:
            marker, marker_color = "!", p['danger']
        else:
            marker, marker_color = "gerando…", p['pending']
        painter.setPen(QColor(marker_color))
        painter.drawText(
            rect.adjusted(58, 0, -rect.width() + 112, 0),
            Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
            marker,
        )

        text = str(index.data(Qt.ItemDataRole.DisplayRole) or "")
        text_rect = rect.adjusted(112, 7, -42, -7)
        document = self._document(text, option.font, text_rect.width())
        text_height = document.size().height()
        text_top = text_rect.top() + max(0.0, (text_rect.height() - text_height) / 2)
        painter.save()
        painter.translate(text_rect.left(), text_top)
        document.drawContents(painter)
        painter.restore()

        if hovered:
            painter.setPen(QColor(p['text_secondary']))
            painter.drawText(
                rect.adjusted(rect.width() - 30, 0, -8, 0),
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                "▶",
            )
        painter.restore()


class ClipboardAudioDialog(QDialog):
    """Janela compacta do atalho, com reprodução progressiva das frases."""

    closed = Signal()
    saved = Signal(str, object)

    def __init__(
        self,
        player: QMediaPlayer,
        output_directory: str,
        parent: QWidget | None = None,
        *,
        phrases: list[str] | None = None,
        recognition_enabled: bool = False,
    ) -> None:
        super().__init__(parent)
        from PySide6.QtCore import Signal

        self.player = player
        self.output_directory = output_directory
        self.temp_dir: Path | None = None
        self.audio_path: Path | None = None
        self.performance: AudioPerformance | None = None
        self.phrase_texts = list(phrases or [])
        self._recognized_segments: list[dict] = []
        self.chunk_paths: list[Path] = []
        self.total_chunks = len(self.phrase_texts)
        self.chunk_paths_by_index: dict[int, Path] = {}
        self.chunk_durations: dict[int, float] = {}
        self.next_chunk_index = 0
        self.current_chunk_path: Path | None = None
        self.current_chunk_index: int | None = None
        self.active_phrase_index: int | None = None
        self.generation_finished = False
        self.play_requested = False
        self.waiting_for_chunk = False
        self.playing_final_audio = False
        self.playing_phrase = False
        self.playing_timed_segment = False
        self._active_segment_end_ms: int | None = None
        self._pending_segment_start_ms: int | None = None
        self._seeking = False
        self.is_closing = False
        self.failure_details = ""
        self._settings = QSettings("g4b3r-desktop", "JoeAudioStudio")

        self.setWindowTitle("Joe — Áudio do atalho")
        self.setWindowFlags(Qt.WindowType.Window)
        self.setWindowIcon(QApplication.windowIcon())
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setMinimumSize(620, 440)
        self.setSizeGripEnabled(True)
        saved_geometry = self._settings.value("clipboard_dialog/geometry")
        if saved_geometry and self.restoreGeometry(saved_geometry):
            pass
        else:
            self.resize(820, 640)

        self.setStyleSheet(_dialog_stylesheet())
        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 12)
        root.setSpacing(8)

        header = QWidget(self)
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(5)
        title_row = QHBoxLayout()
        title_row.setContentsMargins(0, 0, 0, 0)
        title_row.setSpacing(9)
        brand = QLabel("JOE")
        brand.setObjectName("brand")
        title = QLabel("Áudio do atalho")
        title.setObjectName("dialogTitle")
        title_row.addWidget(brand, 0, Qt.AlignmentFlag.AlignVCenter)
        title_row.addWidget(title, 0, Qt.AlignmentFlag.AlignVCenter)
        title_row.addStretch(1)
        header_layout.addLayout(title_row)

        self.preview_label = QLabel("Texto copiado")
        self.preview_label.setObjectName("preview")
        self.preview_label.setWordWrap(False)
        self.preview_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_label.setToolTip("Texto original recebido pelo atalho")
        header_layout.addWidget(self.preview_label)
        root.addWidget(header)

        summary_row = QHBoxLayout()
        summary_row.setContentsMargins(0, 2, 0, 0)
        self.summary_label = QLabel("Preparando frases…")
        self.summary_label.setObjectName("summary")
        self.summary_label.setWordWrap(True)
        summary_row.addWidget(self.summary_label, 1)

        self.details_toggle = QToolButton()
        self.details_toggle.setText("Detalhes")
        self.details_toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.details_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.details_toggle.setCheckable(True)
        self.details_toggle.toggled.connect(self._toggle_details)
        summary_row.addWidget(self.details_toggle, 0)
        self.error_button = QToolButton()
        self.error_button.setText("Ver erro")
        self.error_button.setVisible(False)
        self.error_button.clicked.connect(self._show_failure_details)
        summary_row.addWidget(self.error_button, 0)
        root.addLayout(summary_row)

        self.details_panel = QWidget()
        details_layout = QVBoxLayout(self.details_panel)
        details_layout.setContentsMargins(0, 2, 0, 4)
        details_grid = QGridLayout()
        details_grid.setContentsMargins(0, 0, 0, 0)
        details_grid.setHorizontalSpacing(22)
        details_grid.setVerticalSpacing(5)
        self.detail_values: dict[str, QLabel] = {}
        detail_specs = (
            ("generation", "Tempo de geração"),
            ("duration", "Duração do áudio"),
            ("speed", "Velocidade"),
            ("rate", "Caracteres por segundo"),
        )
        for idx, (key, caption) in enumerate(detail_specs):
            caption_label = QLabel(caption)
            caption_label.setObjectName("detailCaption")
            value_label = QLabel("—")
            value_label.setObjectName("detailValue")
            details_grid.addWidget(caption_label, (idx // 2) * 2, idx % 2)
            details_grid.addWidget(value_label, (idx // 2) * 2 + 1, idx % 2)
            self.detail_values[key] = value_label
        details_layout.addLayout(details_grid)
        self.detail_note = QLabel("")
        self.detail_note.setObjectName("detailCaption")
        self.detail_note.setWordWrap(True)
        details_layout.addWidget(self.detail_note)
        self.details_panel.setVisible(False)
        root.addWidget(self.details_panel)

        self.status_label = QLabel("Aguardando a geração dos primeiros trechos…")
        self.status_label.setObjectName("status")
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.phrase_list = QListWidget(self)
        self.phrase_list.setItemDelegate(PhraseItemDelegate(self.phrase_list))
        self.phrase_list.setWordWrap(True)
        self.phrase_list.setTextElideMode(Qt.TextElideMode.ElideNone)
        self.phrase_list.setMouseTracking(True)
        self.phrase_list.viewport().setMouseTracking(True)
        self.phrase_list.setSpacing(0)
        self.phrase_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.phrase_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.phrase_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.phrase_list.itemClicked.connect(self._play_phrase_item)
        self.phrase_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.phrase_list.customContextMenuRequested.connect(self._show_phrase_context_menu)
        for index, phrase in enumerate(self.phrase_texts, start=1):
            item = QListWidgetItem(phrase)
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setData(Qt.ItemDataRole.UserRole + 1, False)
            item.setData(Qt.ItemDataRole.UserRole + 2, False)
            item.setData(Qt.ItemDataRole.UserRole + 3, False)
            item.setToolTip("Aguardando geração desta frase…")
            self.phrase_list.addItem(item)
        if not self.phrase_texts:
            self.phrase_list.hide()
        root.addWidget(self.phrase_list, 1)

        self.recognition_toggle = QToolButton()
        self.recognition_toggle.setText("Reconhecimento de fala")
        self.recognition_toggle.setArrowType(Qt.ArrowType.DownArrow)
        self.recognition_toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self.recognition_toggle.setCheckable(True)
        self.recognition_toggle.setVisible(bool(recognition_enabled))
        self.recognition_toggle.toggled.connect(self._toggle_recognition)
        root.addWidget(self.recognition_toggle)

        self.recognition_panel = QWidget()
        recognition_layout = QVBoxLayout(self.recognition_panel)
        recognition_layout.setContentsMargins(0, 0, 0, 2)
        recognition_layout.setSpacing(4)
        self.recognition_info = QLabel(
            "O reconhecimento será iniciado quando o WAV completo estiver pronto."
        )
        self.recognition_info.setObjectName("detailCaption")
        self.recognition_info.setWordWrap(True)
        self.recognition_list = QListWidget()
        self.recognition_list.setMaximumHeight(150)
        self.recognition_list.setWordWrap(True)
        self.recognition_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.recognition_list.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.recognition_list.itemClicked.connect(self._play_recognized_segment)
        self.recognition_text_label = QLabel("")
        self.recognition_text_label.setObjectName("detailCaption")
        self.recognition_text_label.setWordWrap(True)
        self.recognition_text_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.recognition_text_label.hide()
        recognition_layout.addWidget(self.recognition_info)
        recognition_layout.addWidget(self.recognition_list)
        recognition_layout.addWidget(self.recognition_text_label)
        self.recognition_panel.setVisible(False)
        root.addWidget(self.recognition_panel)

        player_row = QHBoxLayout()
        player_row.setContentsMargins(0, 4, 0, 0)
        player_row.setSpacing(9)
        self.play_button = QPushButton("▶")
        self.play_button.setObjectName("playButton")
        self.play_button.setToolTip("Reproduzir (Espaço)")
        self.play_button.setEnabled(bool(self.phrase_texts))
        self.play_button.clicked.connect(self._play)
        player_row.addWidget(self.play_button, 0)

        self.seek_slider = ClickableSeekSlider(Qt.Orientation.Horizontal)
        self.seek_slider.setRange(0, 0)
        self.seek_slider.setEnabled(False)
        self.seek_slider.setTracking(False)
        self.seek_slider.sliderPressed.connect(lambda: setattr(self, "_seeking", True))
        self.seek_slider.sliderMoved.connect(self._preview_seek)
        self.seek_slider.sliderReleased.connect(self._commit_seek)
        player_row.addWidget(self.seek_slider, 1)

        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setObjectName("detailCaption")
        self.time_label.setMinimumWidth(82)
        player_row.addWidget(self.time_label, 0)

        self.speed_combo = QComboBox()
        for value, label in (
            (0.75, "0,75×"),
            (1.0, "1×"),
            (1.25, "1,25×"),
            (1.5, "1,5×"),
        ):
            self.speed_combo.addItem(label, value)
        self.speed_combo.setCurrentIndex(1)
        self.speed_combo.setToolTip(
            "O Qt Multimedia altera a taxa de reprodução, mas não garante preservação do tom. "
            "Velocidades diferentes podem mudar o tom da voz."
        )
        self.speed_combo.currentIndexChanged.connect(self._set_playback_rate)
        player_row.addWidget(self.speed_combo, 0)

        self.save_button = QPushButton("↓")
        self.save_button.setObjectName("saveButton")
        self.save_button.setToolTip("Salvar áudio completo em WAV")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save)
        player_row.addWidget(self.save_button, 0)
        root.addLayout(player_row)

        self.player.mediaStatusChanged.connect(self._on_media_status_changed)
        self.player.positionChanged.connect(self._stop_timed_segment)
        self.player.positionChanged.connect(self._on_position_changed)
        self.player.durationChanged.connect(self._on_duration_changed)
        self.player.playbackStateChanged.connect(self._update_play_button)
        self.player.setPlaybackRate(1.0)

        self._shortcuts = []
        self._add_shortcut("Space", self._toggle_play_pause)
        self._add_shortcut("Left", lambda: self._navigate_phrase(-1))
        self._add_shortcut("Right", lambda: self._navigate_phrase(1))
        self._add_shortcut("Return", self._play_selected_phrase)
        self._add_shortcut("Enter", self._play_selected_phrase)
        self._add_shortcut("Escape", self.close)

    def _add_shortcut(self, key: str, callback) -> None:
        shortcut = QShortcut(QKeySequence(key), self)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(callback)
        self._shortcuts.append(shortcut)

    def set_text_preview(self, text: str) -> None:
        normalized = " ".join(str(text or "").split())
        preview = normalized[:80].rstrip()
        if len(normalized) > 80:
            preview += "…"
        self.preview_label.setText(preview or "Texto copiado")
        self.preview_label.setToolTip(normalized)

    def set_progress(self, message: str) -> None:
        if self.is_closing:
            return
        self.status_label.setText(message)

    def _update_summary_progress(self) -> None:
        total = len(self.phrase_texts)
        ready = len(self.chunk_paths_by_index)
        if total:
            self.summary_label.setText(
                f"{ready} de {total} frases prontas · "
                + ("geração concluída" if self.generation_finished else "gerando…")
            )

    def add_chunk(self, index: int, audio_path: Path, total: int, text: str) -> None:
        if self.is_closing:
            return
        self.chunk_paths_by_index[index] = audio_path
        self.chunk_paths.append(audio_path)
        self.chunk_durations[index] = wav_duration_seconds(audio_path)
        if 1 <= index <= self.phrase_list.count():
            item = self.phrase_list.item(index - 1)
            item.setData(Qt.ItemDataRole.UserRole + 1, True)
            item.setData(Qt.ItemDataRole.UserRole + 3, False)
            item.setToolTip("Clique para ouvir somente esta frase.")
            self.phrase_list.viewport().update()
        self.play_button.setEnabled(True)
        self._update_summary_progress()
        if (
            not self.generation_finished
            and not self.play_requested
            and not self.playing_phrase
            and not self.playing_timed_segment
            and not self.playing_final_audio
        ):
            self.status_label.setText(f"Frase {index} pronta · gerando as próximas…")
        if self.waiting_for_chunk and self.play_requested:
            self._play_next_chunk()

    def set_ready(self, result: dict) -> None:
        self.audio_path = Path(result["path"])
        self.generation_finished = True
        performance = result.get("performance")
        self.performance = performance if isinstance(performance, AudioPerformance) else None
        self.save_button.setEnabled(bool(self.audio_path.is_file()))
        self._update_summary_progress()
        if self.performance is not None:
            elapsed = max(0.0, float(self.performance.elapsed_seconds))
            duration = max(0.0, float(self.performance.audio_duration_seconds))
            characters = max(0, int(self.performance.character_count))
            self.detail_values["generation"].setText(format_duration(elapsed))
            self.detail_values["duration"].setText(format_duration(duration))
            if elapsed > 0 and duration > 0:
                ratio = elapsed / duration
                if ratio > 1.05:
                    speed_text = f"{ratio:.1f}× mais lento que o tempo real"
                elif ratio < 0.95:
                    speed_text = f"{1 / ratio:.1f}× mais rápido que o tempo real"
                else:
                    speed_text = "aproximadamente em tempo real"
                self.detail_values["speed"].setText(speed_text.replace(".", ","))
                chars_rate = characters / elapsed if characters else 0.0
                self.detail_values["rate"].setText(
                    f"{chars_rate:.1f} caracteres/s".replace(".", ",")
                )
            else:
                self.detail_values["speed"].setText("Indisponível")
                self.detail_values["rate"].setText("Indisponível")
            self.detail_note.setText(
                "Modo remoto: inclui envio, fila e espera pela API."
                if self.performance.remote
                else "Modo local: pode incluir o carregamento do modelo."
            )
            if elapsed > 0 and duration > 0:
                summary = (
                    f"{duration:.1f} s de áudio · gerado em "
                    f"{self._format_elapsed(elapsed)} · "
                    f"{self.total_chunks or len(self.phrase_texts)} frases"
                )
                self.summary_label.setText(summary.replace(".", ","))
            else:
                self.summary_label.setText(
                    f"Áudio pronto · {self.total_chunks or len(self.phrase_texts)} frases"
                )
        else:
            self.summary_label.setText(
                f"Áudio pronto · {len(self.phrase_texts)} frases"
            )
            self.detail_note.setText(
                "Não foi possível calcular todas as métricas desta geração."
            )
        if self.waiting_for_chunk and self.play_requested:
            self._play_next_chunk()
        elif not (self.playing_phrase or self.playing_final_audio):
            self.status_label.setText("Áudio completo pronto para ouvir ou salvar.")

    @staticmethod
    def _format_elapsed(seconds: float) -> str:
        total = max(0, int(round(seconds)))
        minutes, remainder = divmod(total, 60)
        if minutes >= 60:
            hours, minutes = divmod(minutes, 60)
            return f"{hours}:{minutes:02d}:{remainder:02d}"
        if minutes:
            return f"{minutes}:{remainder:02d}"
        return f"{remainder} s"

    def set_failed(self, message: str, details: str = "") -> None:
        self.failure_details = details.strip()
        self.error_button.setVisible(bool(self.failure_details))
        self.generation_finished = True
        for row in range(self.phrase_list.count()):
            item = self.phrase_list.item(row)
            if not item.data(Qt.ItemDataRole.UserRole + 1):
                item.setData(Qt.ItemDataRole.UserRole + 3, True)
                item.setToolTip("A geração desta frase falhou.")
        self._update_summary_progress()
        self.summary_label.setText(
            f"{len(self.chunk_paths_by_index)} de {self.total_chunks or len(self.phrase_texts)} frases prontas"
        )
        self.status_label.setText(
            f"Falha ao gerar o áudio: {message}. "
            "Os trechos já gerados continuam disponíveis."
        )
        self.phrase_list.viewport().update()
        self.play_button.setEnabled(bool(self.chunk_paths))

    def set_recognition_started(self) -> None:
        if not self.is_closing:
            self.recognition_toggle.setVisible(True)
            self.recognition_info.setText(
                "Reconhecendo o áudio gerado. A primeira execução pode carregar o Parakeet."
            )

    def set_recognition_result(self, result: object) -> None:
        data = result if isinstance(result, dict) else {}
        segments = normalize_timed_segments(list(data.get("segments", [])))
        recognized_text = str(data.get("text", "")).strip()
        backend = str(data.get("backend", "Parakeet Redux"))
        self.recognition_list.clear()

        for index, segment in enumerate(segments):
            start = float(segment["start"])
            end = float(segment["end"])
            item = QListWidgetItem(
                f"{index + 1:03d}  "
                f"[{self._format_time_ms(start * 1000)} – {self._format_time_ms(end * 1000)}]  "
                f"{segment['text']}"
            )
            item.setData(Qt.ItemDataRole.UserRole, index)
            item.setToolTip("Clique para ouvir somente este segmento reconhecido.")
            self.recognition_list.addItem(item)

        self._recognized_segments = segments
        self.recognition_list.setEnabled(bool(segments))
        note = str(data.get("note", "")).strip()
        if segments:
            self.recognition_info.setText(
                f"{backend}: {len(segments)} segmentos com tempo. "
                "Clique em um para ouvir somente esse intervalo."
            )
        else:
            self.recognition_info.setText(
                f"{backend}: transcrição concluída, mas sem tempos de segmento. "
                "Use as frases geradas acima."
            )
        if note:
            self.recognition_info.setToolTip(note)
        if recognized_text:
            self.recognition_text_label.setText(recognized_text)
            self.recognition_text_label.show()
        self.recognition_toggle.setText(
            f"Reconhecimento de fala ({len(segments)} segmentos)"
        )
        self.recognition_toggle.setVisible(True)

    def set_recognition_failed(self, message: str) -> None:
        if not self.is_closing:
            self.recognition_toggle.setVisible(True)
            self.recognition_info.setText(f"Falha no reconhecimento: {message}")

    @staticmethod
    def _format_time_ms(milliseconds: float) -> str:
        total = max(0, int(round(milliseconds / 1000)))
        minutes, seconds = divmod(total, 60)
        hours, minutes = divmod(minutes, 60)
        return f"{hours}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes}:{seconds:02d}"

    def _toggle_details(self, expanded: bool) -> None:
        self.details_panel.setVisible(expanded)
        self.details_toggle.setArrowType(
            Qt.ArrowType.UpArrow if expanded else Qt.ArrowType.DownArrow
        )

    def _toggle_recognition(self, expanded: bool) -> None:
        self.recognition_panel.setVisible(expanded)
        self.recognition_toggle.setArrowType(
            Qt.ArrowType.UpArrow if expanded else Qt.ArrowType.DownArrow
        )

    def _set_active_phrase(self, index: int | None, *, scroll: bool = True) -> None:
        self.active_phrase_index = index
        for row in range(self.phrase_list.count()):
            item = self.phrase_list.item(row)
            item.setData(
                Qt.ItemDataRole.UserRole + 2,
                index is not None and row + 1 == index,
            )
        if index is not None and 1 <= index <= self.phrase_list.count():
            item = self.phrase_list.item(index - 1)
            self.phrase_list.setCurrentItem(item)
            if scroll:
                self.phrase_list.scrollToItem(
                    item, QListWidget.ScrollHint.EnsureVisible
                )
        self.phrase_list.viewport().update()

    def _play_phrase_item(self, item: QListWidgetItem) -> None:
        if self.is_closing:
            return
        index = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(index, int):
            return
        path = self.chunk_paths_by_index.get(index)
        if path is None or not path.is_file():
            self.status_label.setText(
                f"Frase {index} ainda está sendo gerada. Aguarde…"
            )
            return
        self.play_requested = False
        self.waiting_for_chunk = False
        self.playing_final_audio = False
        self.playing_timed_segment = False
        self._active_segment_end_ms = None
        self._pending_segment_start_ms = None
        self.playing_phrase = True
        self.current_chunk_path = path
        self.current_chunk_index = index
        self._set_active_phrase(index)
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        self.player.play()
        self.status_label.setText(
            f"Reproduzindo frase {index} de {len(self.phrase_texts)}."
        )
        self._update_play_button()

    def _play_selected_phrase(self) -> None:
        item = self.phrase_list.currentItem()
        if item is not None:
            self._play_phrase_item(item)
        else:
            self._toggle_play_pause()

    def _navigate_phrase(self, direction: int) -> None:
        if not self.phrase_texts:
            return
        row = self.phrase_list.currentRow()
        if row < 0 and self.active_phrase_index is not None:
            row = self.active_phrase_index - 1
        if row < 0:
            row = 0 if direction > 0 else len(self.phrase_texts) - 1
        else:
            row += direction
        while 0 <= row < self.phrase_list.count():
            item = self.phrase_list.item(row)
            index = int(item.data(Qt.ItemDataRole.UserRole) or 0)
            if index in self.chunk_paths_by_index:
                self.phrase_list.setCurrentItem(item)
                self._play_phrase_item(item)
                return
            row += direction

    def _show_phrase_context_menu(self, position) -> None:
        item = self.phrase_list.itemAt(position)
        if item is None:
            return
        self.phrase_list.setCurrentItem(item)
        index = int(item.data(Qt.ItemDataRole.UserRole) or 0)
        phrase = self.phrase_texts[index - 1] if 1 <= index <= len(self.phrase_texts) else ""
        menu = QMenu(self)
        copy_action = menu.addAction("Copiar texto")
        play_action = menu.addAction("Tocar daqui até o fim")
        ready = index in self.chunk_paths_by_index
        play_action.setEnabled(ready)
        selected = menu.exec(self.phrase_list.mapToGlobal(position))
        if selected is copy_action:
            QApplication.clipboard().setText(phrase)
            self.status_label.setText(f"Texto da frase {index} copiado.")
        elif selected is play_action:
            self._play_from_index(index)

    def _play_from_index(self, index: int) -> None:
        if index not in self.chunk_paths_by_index:
            return
        self.player.stop()
        self.playing_phrase = False
        self.playing_final_audio = False
        self.playing_timed_segment = False
        self._active_segment_end_ms = None
        self._pending_segment_start_ms = None
        self.play_requested = True
        self.waiting_for_chunk = False
        self.next_chunk_index = index - 1
        self._set_active_phrase(index)
        self._play_next_chunk()
        self._update_play_button()

    def _show_failure_details(self) -> None:
        if not self.failure_details:
            return
        message = QMessageBox(self)
        message.setIcon(QMessageBox.Icon.Warning)
        message.setWindowTitle("Detalhes técnicos da falha")
        message.setText("A geração de áudio foi interrompida. Consulte o diagnóstico abaixo.")
        message.setDetailedText(self.failure_details)
        message.exec()

    def _play(self) -> None:
        if self.is_closing:
            return
        state = self.player.playbackState()
        if state == QMediaPlayer.PlaybackState.PlayingState or (
            self.play_requested and self.waiting_for_chunk
        ):
            self._pause()
            return
        if state == QMediaPlayer.PlaybackState.PausedState:
            if not (
                self.playing_final_audio
                or self.playing_phrase
                or self.playing_timed_segment
            ):
                self.play_requested = True
            self.player.play()
            self._update_play_button()
            return
        if self.waiting_for_chunk:
            self.play_requested = True
            self._play_next_chunk()
            self._update_play_button()
            return
        if self.generation_finished and self.audio_path and self.audio_path.is_file():
            self.play_requested = False
            self.waiting_for_chunk = False
            self.playing_final_audio = True
            self.playing_phrase = False
            self.playing_timed_segment = False
            self._active_segment_end_ms = None
            self._pending_segment_start_ms = None
            self.current_chunk_path = None
            self.current_chunk_index = None
            self.player.setSource(QUrl.fromLocalFile(str(self.audio_path.resolve())))
            self.player.play()
            self.status_label.setText("Reproduzindo o áudio completo.")
            self._update_play_button()
            return
        if not self.phrase_texts and not self.chunk_paths:
            return
        self.play_requested = True
        self.next_chunk_index = 0
        self.playing_phrase = False
        self.playing_final_audio = False
        self.playing_timed_segment = False
        self._active_segment_end_ms = None
        self._set_active_phrase(None, scroll=False)
        self._play_next_chunk()
        self._update_play_button()

    def _toggle_play_pause(self) -> None:
        self._play()

    def _pause(self) -> None:
        self.play_requested = False
        self.player.pause()
        self._update_play_button()

    def _play_next_chunk(self) -> None:
        if not self.play_requested or self.is_closing:
            return
        self.playing_phrase = False
        self.playing_timed_segment = False
        self._active_segment_end_ms = None
        next_index = self.next_chunk_index + 1
        path = self.chunk_paths_by_index.get(next_index)
        if path is not None and path.is_file():
            self.next_chunk_index = next_index
            self.current_chunk_index = next_index
            self.current_chunk_path = path
            self.waiting_for_chunk = False
            self._set_active_phrase(next_index)
            self.player.setSource(QUrl.fromLocalFile(str(path.resolve())))
            self.player.play()
            self.status_label.setText(
                f"Reproduzindo frase {next_index}; as próximas continuam sendo geradas…"
                if not self.generation_finished
                else f"Reproduzindo frase {next_index} de {len(self.phrase_texts)}."
            )
            self._update_play_button()
            return

        if self.generation_finished:
            self.play_requested = False
            self.waiting_for_chunk = False
            self.status_label.setText("Reprodução concluída.")
            self._update_play_button()
            return

        self.waiting_for_chunk = True
        self.status_label.setText(
            "Aguardando a próxima frase ficar pronta para continuar…"
        )
        self.player.stop()
        self.player.setSource(QUrl())
        self._update_play_button()

    def _play_recognized_segment(self, item: QListWidgetItem) -> None:
        index = item.data(Qt.ItemDataRole.UserRole)
        if not isinstance(index, int) or not 0 <= index < len(self._recognized_segments):
            return
        if not self.audio_path or not self.audio_path.is_file():
            self.recognition_info.setText("O WAV completo não está mais disponível.")
            return
        segment = self._recognized_segments[index]
        start_ms = max(0, int(round(float(segment["start"]) * 1000)))
        end_ms = max(start_ms + 1, int(round(float(segment["end"]) * 1000)))
        self.play_requested = False
        self.waiting_for_chunk = False
        self.playing_phrase = False
        self.playing_final_audio = False
        self.playing_timed_segment = True
        self._set_active_phrase(None, scroll=False)
        self._active_segment_end_ms = end_ms
        target = str(self.audio_path.resolve())
        current = self.player.source().toLocalFile()
        if not current or Path(current).resolve() != Path(target):
            self._pending_segment_start_ms = start_ms
            self.player.stop()
            self.player.setSource(QUrl.fromLocalFile(target))
        else:
            self._pending_segment_start_ms = None
            self.player.setPosition(start_ms)
            self.player.play()
        self.status_label.setText(
            f"Reproduzindo segmento {index + 1}: {segment['text']}"
        )
        self._update_play_button()

    def _stop_timed_segment(self, position_ms: int) -> None:
        if (
            self.playing_timed_segment
            and self._active_segment_end_ms is not None
            and int(position_ms) >= self._active_segment_end_ms
        ):
            self._active_segment_end_ms = None
            self._pending_segment_start_ms = None
            self.playing_timed_segment = False
            self.player.stop()
            self.status_label.setText("Segmento reconhecido concluído.")
            self._update_play_button()

    def _on_media_status_changed(self, status) -> None:
        if (
            status == QMediaPlayer.MediaStatus.LoadedMedia
            and self._pending_segment_start_ms is not None
            and self.playing_timed_segment
        ):
            start_ms = self._pending_segment_start_ms
            self._pending_segment_start_ms = None
            self.player.setPosition(start_ms)
            self.player.play()
            self._update_play_button()
            return
        if status != QMediaPlayer.MediaStatus.EndOfMedia:
            return
        if self.playing_timed_segment:
            self._active_segment_end_ms = None
            self.playing_timed_segment = False
            self.status_label.setText("Segmento reconhecido concluído.")
        elif self.playing_phrase:
            self.playing_phrase = False
            self.status_label.setText("Frase concluída.")
        elif self.playing_final_audio:
            self.playing_final_audio = False
            self.status_label.setText("Reprodução concluída.")
        elif self.play_requested:
            self._play_next_chunk()
        self._update_play_button()

    def _update_play_button(self, *_args) -> None:
        state = self.player.playbackState()
        active = (
            state == QMediaPlayer.PlaybackState.PlayingState
            or (self.play_requested and self.waiting_for_chunk)
        )
        self.play_button.setText("Ⅱ" if active else "▶")
        self.play_button.setToolTip(
            "Pausar (Espaço)" if active else "Reproduzir (Espaço)"
        )
        self.play_button.setEnabled(
            bool(self.phrase_texts or self.chunk_paths)
        )

    def _on_duration_changed(self, duration_ms: int) -> None:
        duration = max(0, int(duration_ms))
        self.seek_slider.setRange(0, duration)
        self.seek_slider.setEnabled(duration > 0)
        self._update_position(self.player.position())

    def _on_position_changed(self, position_ms: int) -> None:
        self._update_position(position_ms)
        if self.playing_final_audio:
            self._sync_active_phrase_to_position(position_ms)

    def _update_position(self, position_ms: int) -> None:
        duration = max(0, int(self.player.duration()))
        position = min(max(0, int(position_ms)), duration) if duration else 0
        if not self._seeking:
            self.seek_slider.setValue(position)
        self.time_label.setText(
            f"{self._format_time_ms(position)} / {self._format_time_ms(duration)}"
        )

    def _preview_seek(self, position_ms: int) -> None:
        duration = max(0, int(self.player.duration()))
        self.time_label.setText(
            f"{self._format_time_ms(position_ms)} / {self._format_time_ms(duration)}"
        )

    def _commit_seek(self) -> None:
        self.player.setPosition(self.seek_slider.value())
        self._seeking = False
        self._update_position(self.player.position())

    def _sync_active_phrase_to_position(self, position_ms: int) -> None:
        seconds = max(0.0, position_ms / 1000.0)
        cursor = 0.0
        for index in range(1, len(self.phrase_texts) + 1):
            duration = self.chunk_durations.get(index)
            if duration is None:
                continue
            start = cursor
            end = start + duration
            if start <= seconds < end:
                if self.active_phrase_index != index:
                    self._set_active_phrase(index)
                return
            cursor = end + (0.06 if index < len(self.phrase_texts) else 0.0)
        if self.phrase_texts and seconds >= cursor and self.active_phrase_index != len(self.phrase_texts):
            self._set_active_phrase(len(self.phrase_texts))

    def _set_playback_rate(self, *_args) -> None:
        rate = float(self.speed_combo.currentData() or 1.0)
        self.player.setPlaybackRate(rate)
        if rate == 1.0:
            self.speed_combo.setToolTip("Velocidade normal de reprodução.")
        else:
            self.speed_combo.setToolTip(
                "Velocidade ajustada pelo Qt Multimedia. A preservação do tom não é garantida; "
                "a voz pode ficar mais aguda ou mais grave."
            )

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
        self.status_label.setText(f"Cópia salva em: {destination}")
        self.saved.emit(str(destination), self.performance)

    def closeEvent(self, event) -> None:
        self.is_closing = True
        self._settings.setValue("clipboard_dialog/geometry", self.saveGeometry())
        self.play_requested = False
        self._active_segment_end_ms = None
        self._pending_segment_start_ms = None
        try:
            self.player.positionChanged.disconnect(self._stop_timed_segment)
            self.player.positionChanged.disconnect(self._on_position_changed)
            self.player.mediaStatusChanged.disconnect(self._on_media_status_changed)
            self.player.durationChanged.disconnect(self._on_duration_changed)
            self.player.playbackStateChanged.disconnect(self._update_play_button)
        except (RuntimeError, TypeError):
            pass
        current_source = self.player.source().toLocalFile()
        chunk_sources = {str(path.resolve()) for path in self.chunk_paths}
        if current_source and (
            current_source in chunk_sources
            or (self.audio_path and current_source == str(self.audio_path.resolve()))
        ):
            self.player.stop()
            self.player.setSource(QUrl())
        self.player.setPlaybackRate(1.0)
        self.closed.emit()
        super().closeEvent(event)
