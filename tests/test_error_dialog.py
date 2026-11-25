from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtWidgets import QApplication

from joe.ui.error_dialog import ErrorDetailsDialog


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def test_error_dialog_displays_full_text_and_wraps_lines(qapp):
    details = "Falha ao gerar áudio.\n" + ("traceback muito longo " * 80)
    dialog = ErrorDetailsDialog("Erro", details)

    assert dialog.details.toPlainText() == details
    assert dialog.details.isReadOnly()
    assert dialog.details.lineWrapMode() == dialog.details.LineWrapMode.WidgetWidth
    assert dialog.minimumWidth() >= 560

    dialog.close()


def test_copy_all_copies_entire_error_not_only_visible_text(qapp):
    details = "Resumo\n" + ("linha técnica\n" * 500)
    dialog = ErrorDetailsDialog("Erro", details)

    dialog.copy_all()

    assert QApplication.clipboard().text() == details
    dialog.close()
