from __future__ import annotations

from collections.abc import Callable
from typing import Any

from PySide6.QtWidgets import (
    QComboBox, QDialog, QFormLayout, QHBoxLayout, QLabel,
    QMessageBox, QPushButton, QVBoxLayout, QInputDialog,
)


class VoiceProfilesDialog(QDialog):
    """Manage named snapshots of voice/language/pronunciation settings."""

    def __init__(
        self,
        profiles: dict[str, dict[str, Any]],
        active_name: str,
        on_apply: Callable[[dict[str, Any]], None],
        on_save: Callable[[str], dict[str, Any]],
        on_update: Callable[[str], dict[str, Any]],
        on_delete: Callable[[str], None],
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.profiles = dict(profiles)
        self.active_name = active_name
        self.on_apply = on_apply
        self.on_save = on_save
        self.on_update = on_update
        self.on_delete = on_delete
        self.setWindowTitle("Perfis de voz")
        self.resize(520, 250)

        layout = QVBoxLayout(self)
        intro = QLabel(
            "Salve combinações de voz, idioma e pronúncia para reutilizá-las com um clique."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = QFormLayout()
        self.profile_combo = QComboBox()
        self.profile_combo.currentTextChanged.connect(self._selection_changed)
        form.addRow("Perfil salvo", self.profile_combo)
        layout.addLayout(form)

        actions = QHBoxLayout()
        self.apply_button = QPushButton("Usar perfil")
        self.apply_button.clicked.connect(self._apply_selected)
        self.save_button = QPushButton("Salvar atual como…")
        self.save_button.clicked.connect(self._save_new)
        self.update_button = QPushButton("Atualizar perfil")
        self.update_button.clicked.connect(self._update_selected)
        self.delete_button = QPushButton("Excluir")
        self.delete_button.clicked.connect(self._delete_selected)
        for button in (self.apply_button, self.save_button, self.update_button, self.delete_button):
            actions.addWidget(button)
        layout.addLayout(actions)

        close_button = QPushButton("Fechar")
        close_button.clicked.connect(self.accept)
        layout.addWidget(close_button)
        self._refresh()

    def _refresh(self, selected: str = "") -> None:
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        self.profile_combo.addItems(sorted(self.profiles, key=str.casefold))
        index = self.profile_combo.findText(selected or self.active_name)
        if index >= 0:
            self.profile_combo.setCurrentIndex(index)
        self.profile_combo.blockSignals(False)
        self._selection_changed(self.profile_combo.currentText())

    def _selection_changed(self, name: str) -> None:
        available = bool(name and name in self.profiles)
        self.apply_button.setEnabled(available)
        self.update_button.setEnabled(available)
        self.delete_button.setEnabled(available)

    def _apply_selected(self) -> None:
        name = self.profile_combo.currentText()
        profile = self.profiles.get(name)
        if profile is None:
            return
        self.on_apply(profile)
        self.active_name = name
        self.accept()

    def _save_new(self) -> None:
        name, accepted = QInputDialog.getText(self, "Salvar perfil", "Nome do novo perfil:")
        name = name.strip()
        if not accepted or not name:
            return
        if any(existing.casefold() == name.casefold() for existing in self.profiles):
            QMessageBox.information(self, "Perfil existente", "Já existe um perfil com esse nome.")
            return
        self.profiles[name] = self.on_save(name)
        self.active_name = name
        self._refresh(name)

    def _update_selected(self) -> None:
        name = self.profile_combo.currentText()
        if name not in self.profiles:
            return
        answer = QMessageBox.question(
            self, "Atualizar perfil",
            f"Substituir “{name}” pelas configurações atuais?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.profiles[name] = self.on_update(name)
        self._refresh(name)

    def _delete_selected(self) -> None:
        name = self.profile_combo.currentText()
        if name not in self.profiles:
            return
        answer = QMessageBox.question(
            self, "Excluir perfil",
            f"Excluir o perfil “{name}”? A voz e os arquivos originais não serão apagados.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        self.on_delete(name)
        self.profiles.pop(name, None)
        if self.active_name == name:
            self.active_name = ""
        self._refresh()
