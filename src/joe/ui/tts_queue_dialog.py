from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
)


class TTSQueueDialog(QDialog):
    """In-memory TTS queue with retry, remove and reorder controls."""

    def __init__(
        self,
        on_start: Callable[[], None],
        on_remove: Callable[[str], None],
        on_move: Callable[[str, int], None],
        on_clear: Callable[[], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.on_start = on_start
        self.on_remove = on_remove
        self.on_move = on_move
        self.on_clear = on_clear
        self.running = False
        self.setWindowTitle("Fila de geração")
        self.resize(760, 480)

        layout = QVBoxLayout(self)
        self.summary = QLabel("A fila fica disponível somente durante esta sessão.")
        layout.addWidget(self.summary)
        self.items = QListWidget()
        self.items.currentItemChanged.connect(self._selection_changed)
        layout.addWidget(self.items, 1)

        actions = QHBoxLayout()
        self.start_button = QPushButton("▶ Iniciar fila")
        self.start_button.clicked.connect(self.on_start)
        self.up_button = QPushButton("↑")
        self.up_button.setToolTip("Mover tarefa para cima")
        self.up_button.clicked.connect(lambda: self._move(-1))
        self.down_button = QPushButton("↓")
        self.down_button.setToolTip("Mover tarefa para baixo")
        self.down_button.clicked.connect(lambda: self._move(1))
        self.remove_button = QPushButton("Remover")
        self.remove_button.clicked.connect(self._remove)
        self.clear_button = QPushButton("Limpar fila")
        self.clear_button.clicked.connect(self._clear)
        close_button = QPushButton("Fechar")
        close_button.clicked.connect(self.hide)
        for button in (
            self.start_button, self.up_button, self.down_button,
            self.remove_button, self.clear_button, close_button
        ):
            actions.addWidget(button)
        layout.addLayout(actions)
        self.refresh([], running=False)

    def refresh(self, entries: list[dict[str, Any]], running: bool) -> None:
        selected_id = self._selected_id()
        self.running = running
        self.items.clear()
        for entry in entries:
            state = str(entry.get("status", "Pendente"))
            title = str(entry.get("title") or "Texto sem título")
            characters = int(entry.get("character_count", 0))
            voice = str(entry.get("voice_alias") or "Voz padrão")
            error = str(entry.get("error", "")).strip()
            label = f"[{state}] {title} • {characters} caracteres • {voice}"
            if error:
                label += f" • {error[:100]}"
            item = QListWidgetItem(label)
            item.setData(Qt.ItemDataRole.UserRole, entry.get("id"))
            self.items.addItem(item)
            if entry.get("id") == selected_id:
                self.items.setCurrentItem(item)
        counts = {
            "pending": sum(e.get("status", "Pendente") == "Pendente" for e in entries),
            "failed": sum(e.get("status") == "Falhou" for e in entries),
            "running": sum(e.get("status") == "Gerando" for e in entries),
        }
        self.summary.setText(
            f"{len(entries)} tarefa(s) • {counts['pending']} pendente(s) • "
            f"{counts['failed']} falha(s) • {counts['running']} em andamento. "
            "A fila não é gravada no disco."
        )
        self._selection_changed(self.items.currentItem(), None)

    def _selected_id(self) -> str | None:
        current = self.items.currentItem()
        return str(current.data(Qt.ItemDataRole.UserRole)) if current else None

    def _selection_changed(self, *_args) -> None:
        row = self.items.currentRow()
        has_selection = row >= 0
        self.start_button.setEnabled(not self.running and self.items.count() > 0)
        self.remove_button.setEnabled(not self.running and has_selection)
        self.up_button.setEnabled(not self.running and row > 0)
        self.down_button.setEnabled(
            not self.running and has_selection and row < self.items.count() - 1
        )
        self.clear_button.setEnabled(not self.running and self.items.count() > 0)

    def _move(self, offset: int) -> None:
        item_id = self._selected_id()
        row = self.items.currentRow()
        target = row + offset
        if item_id and 0 <= target < self.items.count():
            self.on_move(item_id, target)

    def _remove(self) -> None:
        item_id = self._selected_id()
        if item_id:
            self.on_remove(item_id)

    def _clear(self) -> None:
        if not self.items.count():
            return
        answer = QMessageBox.question(
            self,
            "Limpar fila",
            "Remover todas as tarefas pendentes e com falha? Os WAVs já gerados permanecem.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer == QMessageBox.StandardButton.Yes:
            self.on_clear()
