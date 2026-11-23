from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QDate, QObject, QRunnable, QThreadPool, QTimer, QUrl, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QHeaderView,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from joe.audio_history import AudioHistoryStore
from joe.audio_performance import format_duration


class _HistoryLoadSignals(QObject):
    completed = Signal(int, object, object, int, int, object)
    failed = Signal(int, str)


class _HistoryLoadWorker(QRunnable):
    """Read one history page and its filter metadata outside the GUI thread."""

    def __init__(
        self,
        request_id: int,
        store: AudioHistoryStore,
        filters: dict[str, object],
        *,
        limit: int,
        offset: int,
    ) -> None:
        super().__init__()
        self.request_id = request_id
        self.store = store
        self.filters = filters
        self.limit = limit
        self.offset = offset
        self.signals = _HistoryLoadSignals()

    def run(self) -> None:
        try:
            entries = self.store.list_entries(
                **self.filters,
                limit=self.limit,
                offset=self.offset,
            )
            total = self.store.count_entries(**self.filters)
            voices = self.store.available_voices()
            text_count = self.store.count_text_entries()
            missing_ids = {
                entry.id for entry in entries if not Path(entry.audio_path).is_file()
            }
            self.signals.completed.emit(
                self.request_id, entries, voices, total, text_count, missing_ids
            )
        except Exception as exc:  # noqa: BLE001
            self.signals.failed.emit(self.request_id, str(exc))



class AudioHistoryPanel(QWidget):
    """Search, play and manage generated WAV files without moving them on import."""

    status_message = Signal(str)

    def __init__(
        self,
        store: AudioHistoryStore,
        player: QMediaPlayer,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.store = store
        self.player = player
        self.page_size = 100
        self.page_offset = 0
        self.total_entries = 0
        self._load_request_id = 0
        self._active_workers: dict[int, _HistoryLoadWorker] = {}
        self._entries_by_id: dict[int, object] = {}
        self._text_entries_count = 0
        self._loading_table = False
        self._search_timer = QTimer(self)
        self._search_timer.setSingleShot(True)
        self._search_timer.setInterval(250)
        self._search_timer.timeout.connect(self.refresh)
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 22, 24, 22)
        layout.setSpacing(12)

        title = QLabel("Histórico de áudios")
        title.setObjectName("pageTitle")
        description = QLabel(
            "Áudios WAV gerados e salvos pelo Joe. Os arquivos permanecem na pasta original; "
            "o histórico guarda seus metadados e permite pesquisar, reproduzir e organizar."
        )
        description.setObjectName("muted")
        description.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(description)

        filters = QHBoxLayout()
        self.search = QLineEdit()
        self.search.setPlaceholderText("Pesquisar arquivo, voz ou texto salvo…")
        self.search.textChanged.connect(self._filters_changed)
        filters.addWidget(self.search, 2)

        self.voice_filter = QComboBox()
        self.voice_filter.addItem("Todas as vozes", "")
        self.voice_filter.currentIndexChanged.connect(self._filters_changed)
        filters.addWidget(self.voice_filter, 1)
        layout.addLayout(filters)

        advanced_filters = QHBoxLayout()
        self.from_enabled = QCheckBox("De:")
        self.from_date = QDateEdit(QDate.currentDate().addDays(-30))
        self.from_date.setCalendarPopup(True)
        self.from_date.setDisplayFormat("dd/MM/yyyy")
        self.from_date.setEnabled(False)
        self.from_enabled.toggled.connect(self.from_date.setEnabled)
        self.from_enabled.toggled.connect(self._filters_changed)
        self.from_date.dateChanged.connect(self._filters_changed)

        self.to_enabled = QCheckBox("Até:")
        self.to_date = QDateEdit(QDate.currentDate())
        self.to_date.setCalendarPopup(True)
        self.to_date.setDisplayFormat("dd/MM/yyyy")
        self.to_date.setEnabled(False)
        self.to_enabled.toggled.connect(self.to_date.setEnabled)
        self.to_enabled.toggled.connect(self._filters_changed)
        self.to_date.dateChanged.connect(self._filters_changed)

        self.min_duration = QDoubleSpinBox()
        self.min_duration.setRange(0, 999999)
        self.min_duration.setDecimals(1)
        self.min_duration.setSuffix(" s")
        self.min_duration.setPrefix("Mín. ")
        self.min_duration.setSpecialValueText("Mín. qualquer")
        self.min_duration.valueChanged.connect(self._filters_changed)

        self.max_duration = QDoubleSpinBox()
        self.max_duration.setRange(0, 999999)
        self.max_duration.setDecimals(1)
        self.max_duration.setSuffix(" s")
        self.max_duration.setPrefix("Máx. ")
        self.max_duration.setSpecialValueText("Máx. qualquer")
        self.max_duration.valueChanged.connect(self._filters_changed)

        for widget in (
            self.from_enabled,
            self.from_date,
            self.to_enabled,
            self.to_date,
            self.min_duration,
            self.max_duration,
        ):
            advanced_filters.addWidget(widget)
        advanced_filters.addStretch(1)
        layout.addLayout(advanced_filters)

        self.table = QTableWidget(0, 10)
        self.table.setHorizontalHeaderLabels([
            "Arquivo", "Data", "Duração", "Tamanho",
            "Voz", "Modelo TTS", "Tempo de geração", "Velocidade",
            "Configuração TTS", "Origem",
        ])
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        column_widths = (0, 130, 75, 85, 130, 130, 115, 75, 150, 95)
        for column, width in enumerate(column_widths):
            if column:
                self.table.setColumnWidth(column, width)
        self.table.itemDoubleClicked.connect(lambda _item: self.play_selected())
        layout.addWidget(self.table, 1)

        pagination = QHBoxLayout()
        self.previous_page_button = QPushButton("← Anterior")
        self.page_info_label = QLabel("Histórico ainda não carregado")
        self.page_info_label.setObjectName("muted")
        self.next_page_button = QPushButton("Próxima →")
        self.previous_page_button.clicked.connect(self._previous_page)
        self.next_page_button.clicked.connect(self._next_page)
        self.previous_page_button.setEnabled(False)
        self.next_page_button.setEnabled(False)
        pagination.addWidget(self.previous_page_button)
        pagination.addStretch(1)
        pagination.addWidget(self.page_info_label)
        pagination.addStretch(1)
        pagination.addWidget(self.next_page_button)
        layout.addLayout(pagination)

        actions = QHBoxLayout()
        self.play_button = QPushButton("Reproduzir")
        self.open_button = QPushButton("Abrir pasta")
        self.rename_button = QPushButton("Renomear")
        self.export_button = QPushButton("Exportar cópia…")
        self.delete_button = QPushButton("Excluir")
        self.show_text_button = QPushButton("Ver texto salvo")
        self.clear_text_button = QPushButton("Apagar textos salvos")
        self.refresh_button = QPushButton("Atualizar")
        self.play_button.clicked.connect(self.play_selected)
        self.open_button.clicked.connect(self.open_folder)
        self.rename_button.clicked.connect(self.rename_selected)
        self.export_button.clicked.connect(self.export_selected)
        self.delete_button.clicked.connect(self.delete_selected)
        self.show_text_button.clicked.connect(self.show_selected_text)
        self.clear_text_button.clicked.connect(self.clear_stored_texts)
        self.refresh_button.clicked.connect(self.refresh)
        for button in (
            self.play_button,
            self.open_button,
            self.rename_button,
            self.export_button,
            self.delete_button,
            self.show_text_button,
        ):
            actions.addWidget(button)
        actions.addStretch(1)
        actions.addWidget(self.refresh_button)
        layout.addLayout(actions)

        privacy_actions = QHBoxLayout()
        privacy_actions.addStretch(1)
        privacy_actions.addWidget(self.clear_text_button)
        layout.addLayout(privacy_actions)

        self.summary = QLabel("Nenhum áudio registrado ainda.")
        self.summary.setObjectName("muted")
        layout.addWidget(self.summary)

        self.table.itemSelectionChanged.connect(self._update_action_state)
        self._update_action_state()

    @staticmethod
    def _format_generation_settings(settings_json: str) -> str:
        if not settings_json:
            return "—"
        try:
            settings = json.loads(settings_json)
        except (TypeError, ValueError):
            return "Detalhes indisponíveis"
        if not isinstance(settings, dict):
            return "—"
        if settings.get("local_settings_applied") is False:
            return "Remoto"
        device = "CUDA" if settings.get("device") == "cuda" else "CPU"
        try:
            threads = max(1, int(settings.get("cpu_threads", 1)))
        except (TypeError, ValueError):
            threads = 1
        parts = [device, f"{threads} thread(s)"]
        if device == "CPU":
            parts.append("INT8" if settings.get("quantization") else "sem INT8")
        profile_names = {
            "economy": "Econômico",
            "balanced": "Equilibrado",
            "performance": "Desempenho",
            "custom": "Personalizado",
        }
        profile = profile_names.get(str(settings.get("profile", "custom")))
        if profile:
            parts.append(profile)
        return " · ".join(parts)

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        value = max(0, int(size_bytes))
        if value >= 1024 * 1024 * 1024:
            return f"{value / (1024 * 1024 * 1024):.2f} GB"
        if value >= 1024 * 1024:
            return f"{value / (1024 * 1024):.1f} MB"
        if value >= 1024:
            return f"{value / 1024:.1f} KB"
        return f"{value} B"

    def _selected_entry(self):
        if self._loading_table:
            return None
        row = self.table.currentRow()
        if row < 0:
            return None
        item = self.table.item(row, 0)
        if item is None:
            return None
        try:
            entry_id = int(item.data(Qt.ItemDataRole.UserRole))
        except (TypeError, ValueError):
            return None
        return self._entries_by_id.get(entry_id)

    def _update_action_state(self) -> None:
        if self._loading_table:
            return
        entry = self._selected_entry()
        exists = bool(entry and Path(entry.audio_path).is_file())
        self.play_button.setEnabled(exists)
        self.open_button.setEnabled(bool(entry and Path(entry.audio_path).parent.is_dir()))
        self.rename_button.setEnabled(exists)
        self.export_button.setEnabled(exists)
        self.delete_button.setEnabled(bool(entry))
        self.show_text_button.setEnabled(bool(entry and entry.text_content))
        self.clear_text_button.setEnabled(self._text_entries_count > 0)

    def _current_filters(self) -> dict[str, object]:
        return {
            "search": self.search.text(),
            "voice": str(self.voice_filter.currentData() or ""),
            "date_from": (
                self.from_date.date().toString("yyyy-MM-dd")
                if self.from_enabled.isChecked() else ""
            ),
            "date_to": (
                self.to_date.date().toString("yyyy-MM-dd")
                if self.to_enabled.isChecked() else ""
            ),
            "min_duration": self.min_duration.value(),
            "max_duration": self.max_duration.value(),
        }

    def _filters_changed(self, *_args) -> None:
        self.page_offset = 0
        self._search_timer.start()

    def _previous_page(self) -> None:
        if self.page_offset > 0:
            self.page_offset = max(0, self.page_offset - self.page_size)
            self.refresh()

    def _next_page(self) -> None:
        if self.page_offset + self.page_size < self.total_entries:
            self.page_offset += self.page_size
            self.refresh()

    def refresh(self, *_args) -> None:
        self._search_timer.stop()
        self._load_request_id += 1
        request_id = self._load_request_id
        worker = _HistoryLoadWorker(
            request_id,
            self.store,
            self._current_filters(),
            limit=self.page_size,
            offset=self.page_offset,
        )
        worker.signals.completed.connect(self._on_history_loaded)
        worker.signals.failed.connect(self._on_history_load_failed)
        self._active_workers[request_id] = worker
        self.summary.setText("Carregando histórico…")
        self.page_info_label.setText("Consultando banco de dados…")
        self.refresh_button.setEnabled(False)
        QThreadPool.globalInstance().start(worker)

    def _on_history_loaded(
        self,
        request_id: int,
        entries: list,
        voices: list,
        total: int,
        text_count: int,
        missing_ids: set[int],
    ) -> None:
        self._active_workers.pop(request_id, None)
        if request_id != self._load_request_id:
            return

        self.total_entries = total
        if self.page_offset and self.page_offset >= total:
            self.page_offset = max(0, ((max(total, 1) - 1) // self.page_size) * self.page_size)
            self.refresh()
            return

        selected = self._selected_entry()
        selected_id = selected.id if selected is not None else None

        current_voice = self.voice_filter.currentData()
        self.voice_filter.blockSignals(True)
        self.voice_filter.clear()
        self.voice_filter.addItem("Todas as vozes", "")
        for voice in voices:
            self.voice_filter.addItem(voice, voice)
        voice_index = self.voice_filter.findData(current_voice)
        self.voice_filter.setCurrentIndex(max(0, voice_index))
        self.voice_filter.blockSignals(False)
        if current_voice and voice_index < 0:
            self.refresh()
            return

        self._entries_by_id = {entry.id: entry for entry in entries}
        self._text_entries_count = text_count

        self._loading_table = True
        self.table.setUpdatesEnabled(False)
        self.table.setSortingEnabled(False)
        self.table.blockSignals(True)
        self.table.setRowCount(len(entries))
        selected_row = -1

        for row, entry in enumerate(entries):
            try:
                date_label = datetime.fromisoformat(entry.created_at).astimezone().strftime(
                    "%d/%m/%Y %H:%M"
                )
            except ValueError:
                date_label = entry.created_at
            speed = f"{entry.realtime_speed:.2f}×".replace(".", ",")
            display_name = (
                f"{entry.file_name} (arquivo ausente)"
                if entry.id in missing_ids else entry.file_name
            )
            values = (
                display_name,
                date_label,
                format_duration(entry.duration_seconds),
                self._format_size(entry.size_bytes),
                entry.voice_label,
                entry.model_variant or "Padrão",
                format_duration(entry.generation_seconds),
                speed,
                self._format_generation_settings(entry.generation_settings_json),
                entry.source_label,
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, entry.id)
                    item.setToolTip(entry.audio_path)
                elif column == 8 and entry.generation_settings_json:
                    try:
                        settings = json.loads(entry.generation_settings_json)
                        item.setToolTip(
                            json.dumps(settings, ensure_ascii=False, indent=2, sort_keys=True)
                        )
                    except (TypeError, ValueError):
                        item.setToolTip(entry.generation_settings_json)
                self.table.setItem(row, column, item)
            if entry.id == selected_id:
                selected_row = row

        self.table.blockSignals(False)
        self.table.setSortingEnabled(True)
        self.table.setUpdatesEnabled(True)
        self._loading_table = False

        if selected_row >= 0:
            self.table.selectRow(selected_row)

        current_page = (self.page_offset // self.page_size) + 1
        page_count = max(1, (total + self.page_size - 1) // self.page_size)
        first_item = self.page_offset + 1 if total else 0
        last_item = min(self.page_offset + len(entries), total)
        self.page_info_label.setText(
            f"Página {current_page}/{page_count} · {first_item}–{last_item} de {total}"
        )
        missing_count = len(missing_ids)
        suffix = f" {missing_count} arquivo(s) não encontrado(s) nesta página." if missing_count else ""
        self.summary.setText(
            f"{total} áudio(s) no histórico. Exibindo até {self.page_size} por página.{suffix}"
        )
        self.previous_page_button.setEnabled(self.page_offset > 0)
        self.next_page_button.setEnabled(self.page_offset + self.page_size < total)
        self.refresh_button.setEnabled(True)
        self._update_action_state()

    def _on_history_load_failed(self, request_id: int, error: str) -> None:
        self._active_workers.pop(request_id, None)
        if request_id != self._load_request_id:
            return
        self.summary.setText(f"Não foi possível carregar o histórico: {error}")
        self.page_info_label.setText("Falha ao carregar")
        self.refresh_button.setEnabled(True)
        self.previous_page_button.setEnabled(self.page_offset > 0)
        self.next_page_button.setEnabled(self.page_offset + self.page_size < self.total_entries)
        self.status_message.emit(f"Falha ao carregar histórico: {error}")

    def play_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        path = Path(entry.audio_path)
        if not path.is_file():
            QMessageBox.warning(self, "Áudio não encontrado", f"O arquivo não existe mais:\n{path}")
            self.refresh()
            return
        self.player.stop()
        self.player.setSource(QUrl.fromLocalFile(str(path.resolve())))
        self.player.play()
        self.status_message.emit(f"Reproduzindo do histórico: {entry.file_name}")

    def open_folder(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        folder = Path(entry.audio_path).parent
        if folder.is_dir():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def rename_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        new_name, accepted = QInputDialog.getText(
            self, "Renomear áudio", "Novo nome do arquivo:", text=entry.file_name
        )
        if not accepted:
            return
        try:
            current_source = self.player.source().toLocalFile()
            if (
                current_source
                and Path(current_source).resolve() == Path(entry.audio_path).resolve()
            ):
                self.player.stop()
                self.player.setSource(QUrl())
            self.store.rename_audio(entry.id, new_name)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.warning(self, "Não foi possível renomear", str(exc))
            return
        self.refresh()
        self.status_message.emit("Áudio renomeado.")

    def export_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        output, _ = QFileDialog.getSaveFileName(
            self,
            "Exportar cópia do áudio",
            str(Path.home() / entry.file_name),
            "Áudio WAV (*.wav)",
        )
        if not output:
            return
        if not output.lower().endswith(".wav"):
            output += ".wav"
        try:
            self.store.export_copy(entry.id, output)
        except (OSError, ValueError, KeyError) as exc:
            QMessageBox.critical(self, "Falha ao exportar áudio", str(exc))
            return
        self.status_message.emit(f"Cópia exportada: {output}")

    def show_selected_text(self) -> None:
        entry = self._selected_entry()
        if entry is None or not entry.text_content:
            QMessageBox.information(
                self,
                "Texto indisponível",
                "O texto original não foi armazenado para este áudio.",
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle(f"Texto original — {entry.file_name}")
        dialog.resize(720, 520)
        layout = QVBoxLayout(dialog)

        editor = QTextEdit()
        editor.setReadOnly(True)
        editor.setPlainText(entry.text_content)
        layout.addWidget(editor, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        copy_button = QPushButton("Copiar texto")
        copy_button.clicked.connect(
            lambda: QApplication.clipboard().setText(entry.text_content or "")
        )
        buttons.addButton(copy_button, QDialogButtonBox.ButtonRole.ActionRole)
        buttons.rejected.connect(dialog.reject)
        buttons.accepted.connect(dialog.accept)
        layout.addWidget(buttons)
        dialog.exec()

    def clear_stored_texts(self) -> None:
        count = self._text_entries_count
        if count <= 0:
            self.status_message.emit("Não há textos salvos no histórico.")
            return
        answer = QMessageBox.question(
            self,
            "Apagar textos salvos",
            f"Apagar o texto original de {count} áudio(s)? "
            "Os arquivos WAV e os metadados serão mantidos.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        removed = self.store.clear_text_content()
        self.refresh()
        self.status_message.emit(f"Textos apagados: {removed}. Os áudios foram preservados.")

    def delete_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        answer = QMessageBox.question(
            self,
            "Excluir áudio",
            f"Excluir permanentemente o arquivo e o registro do histórico?\n\n"
            f"{entry.file_name}\n{entry.audio_path}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            current_source = self.player.source().toLocalFile()
            if current_source and Path(current_source).resolve() == Path(entry.audio_path).resolve():
                self.player.stop()
                self.player.setSource(QUrl())
            self.store.delete(entry.id, delete_file=True)
        except (OSError, sqlite3.Error) as exc:
            QMessageBox.critical(self, "Falha ao excluir áudio", str(exc))
            return
        self.refresh()
        self.status_message.emit("Áudio excluído do disco e do histórico.")
