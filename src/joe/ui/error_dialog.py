from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QPushButton,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)


class ErrorDetailsDialog(QDialog):
    """Scrollable, resizable error viewer with reliable full-text copy."""

    def __init__(self, title: str, details: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(560, 360)
        self.resize(820, 620)

        layout = QVBoxLayout(self)
        self.details = QTextEdit(self)
        self.details.setReadOnly(True)
        self.details.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.details.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.details.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.details.setPlainText(details)
        layout.addWidget(self.details, 1)

        actions = QHBoxLayout()
        copy_button = QPushButton("Copiar erro completo", self)
        copy_button.clicked.connect(self.copy_all)
        actions.addWidget(copy_button)
        actions.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.reject)
        buttons.accepted.connect(self.accept)
        actions.addWidget(buttons)
        layout.addLayout(actions)

    def copy_all(self) -> None:
        QApplication.clipboard().setText(self.details.toPlainText())
