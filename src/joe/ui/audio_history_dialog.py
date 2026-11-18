from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)

from joe.audio_performance import format_duration


class AudioHistoryDialog(QDialog):
    """Searchable history list; file operations are delegated to the main window."""

    def __init__(
        self,
        entries: list[dict[str, Any]],
        on_play: Callable[[dict[str, Any]], None],
        on_open_folder: Callable[[dict[str, Any]], None],
        on_remove: Callable[[dict[str, Any]], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.entries = list(entries)
        self.on_play = on_play
        self.on_open_folder = on_open_folder
        self.on_remove = on_remove
        self.setWindowTitle("Histórico de áudios")
        self.resize(760, 520)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Áudios recentes gerados pelo Joe. O histórico guarda metadados, "
            "não armazena o texto original."
        ))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Pesquisar por arquivo, voz, idioma ou backend…")
        self.search.textChanged.connect(self._filter)
        layout.addWidget(self.search)

        self.items = QListWidget()
        self.items.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.items, 1)

        actions = QHBoxLayout()
        self.play_button = QPushButton("▶ Reproduzir")
        self.play_button.clicked.connect(self._play_selected)
        self.folder_button = QPushButton("Abrir pasta")
        self.folder_button.clicked.connect(self._open_selected_folder)
        self.remove_button = QPushButton("Remover do histórico")
        self.remove_button.clicked.connect(self._remove_selected)
        close_button = QPushButton("Fechar")
        close_button.clicked.connect(self.accept)
        for button in (
            self.play_button, self.folder_button, self.remove_button, close_button
        ):
            actions.addWidget(button)
        layout.addLayout(actions)
        self._populate()
        self._selection_changed(self.items.currentItem(), None)

    def _display_text(self, entry: dict[str, Any]) -> str:
        path = Path(entry["path"])
        name = path.name or entry["path"]
        duration = format_duration(entry.get("duration_seconds", 0.0))
        created = entry.get("created_at") or "Data desconhecida"
        voice = entry.get("voice_alias") or Path(entry.get("voice_path") or "voz padrão").stem
        backend = entry.get("backend") or "Backend desconhecido"
        missing = " — arquivo não encontrado" if not path.is_file() else ""
        return f"{created}  •  {name}  •  {duration}  •  {voice}  •  {backend}{missing}"

    def _populate(self) -> None:
        self.items.clear()
        for entry in self.entries:
            item = QListWidgetItem(self._display_text(entry))
            item.setData(Qt.ItemDataRole.UserRole, entry)
            self.items.addItem(item)
        if self.items.count():
            self.items.setCurrentRow(0)

    def _filter(self, query: str) -> None:
        query = query.strip().casefold()
        for index in range(self.items.count()):
            item = self.items.item(index)
            entry = item.data(Qt.ItemDataRole.UserRole) or {}
            haystack = " ".join(
                str(entry.get(key, ""))
                for key in ("path", "voice_alias", "voice_path", "language", "backend", "source")
            ).casefold()
            item.setHidden(bool(query) and query not in haystack)

    def _selected_entry(self) -> dict[str, Any] | None:
        item = self.items.currentItem()
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _selection_changed(self, *_args) -> None:
        entry = self._selected_entry()
        exists = bool(entry and Path(entry["path"]).is_file())
        self.play_button.setEnabled(exists)
        self.folder_button.setEnabled(bool(entry and Path(entry["path"]).parent.is_dir()))
        self.remove_button.setEnabled(entry is not None)

    def _play_selected(self) -> None:
        entry = self._selected_entry()
        if entry and Path(entry["path"]).is_file():
            self.on_play(entry)

    def _open_selected_folder(self) -> None:
        entry = self._selected_entry()
        if entry:
            self.on_open_folder(entry)

    def _remove_selected(self) -> None:
        entry = self._selected_entry()
        if entry is None:
            return
        answer = QMessageBox.question(
            self,
            "Remover do histórico",
            "Remover este registro do histórico? O arquivo de áudio não será apagado.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.on_remove(entry)
        self.entries = [
            current for current in self.entries
            if not (
                current.get("path") == entry.get("path")
                and current.get("created_at") == entry.get("created_at")
            )
        ]
        self._populate()
        self._selection_changed(self.items.currentItem(), None)
