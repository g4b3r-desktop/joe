from __future__ import annotations

import os
import wave
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import QSettings, Qt
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import QApplication, QGroupBox, QScrollArea

from joe.audio_performance import AudioPerformance
from joe.ui.clipboard_audio_dialog import ClipboardAudioDialog


@pytest.fixture
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


def write_silent_wav(path: Path, duration: float = 0.5) -> None:
    sample_rate = 22050
    frames = int(sample_rate * duration)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(b"\x00\x00" * frames)


def test_dialog_handles_long_text_and_35_phrases(qapp):
    phrases = [f"Esta é a frase de teste número {index:02d}." for index in range(1, 36)]
    dialog = ClipboardAudioDialog(QMediaPlayer(), "", phrases=phrases)
    original = " ".join(phrases)

    dialog.set_text_preview(original)

    assert dialog.phrase_list.count() == 35
    assert dialog.details_panel.isHidden()
    assert dialog.preview_label.text().endswith("…")
    assert len(dialog.preview_label.text()) <= 81
    assert dialog.findChild(QScrollArea) is None
    assert dialog.findChild(QGroupBox) is None
    assert dialog.phrase_list.verticalScrollBarPolicy() == Qt.ScrollBarPolicy.ScrollBarAsNeeded

    dialog.close()


def test_dialog_handles_short_text_and_original_preview(qapp):
    phrases = ["Primeira frase.", "Segunda frase."]
    dialog = ClipboardAudioDialog(QMediaPlayer(), "", phrases=phrases)

    dialog.set_text_preview("Primeira frase. Segunda frase.")

    assert dialog.phrase_list.count() == 2
    assert dialog.preview_label.text() == "Primeira frase. Segunda frase."
    assert not dialog.details_panel.isVisible()

    dialog.close()


def test_ready_chunk_gets_checkmark_state_and_realtime_factor(qapp, tmp_path):
    phrase_path = tmp_path / "phrase_001.wav"
    final_path = tmp_path / "complete.wav"
    write_silent_wav(phrase_path)
    write_silent_wav(final_path, duration=43.0)

    dialog = ClipboardAudioDialog(
        QMediaPlayer(),
        "",
        phrases=["Uma frase pronta.", "Uma frase pendente."],
    )
    dialog.add_chunk(1, phrase_path, 2, "Uma frase pronta.")

    first = dialog.phrase_list.item(0)
    second = dialog.phrase_list.item(1)
    assert first.data(Qt.ItemDataRole.UserRole + 1) is True
    assert second.data(Qt.ItemDataRole.UserRole + 1) is False

    dialog.set_ready({
        "path": str(final_path),
        "performance": AudioPerformance(
            elapsed_seconds=172.0,
            audio_duration_seconds=43.0,
            character_count=700,
            remote=True,
        ),
    })

    assert dialog.save_button.isEnabled()
    assert "43,0 s de áudio" in dialog.summary_label.text()
    assert "2:52" in dialog.summary_label.text()
    assert "4,0× mais lento que o tempo real" == dialog.detail_values["speed"].text()
    assert "envio, fila" in dialog.detail_note.text()

    dialog.close()

def test_dialog_appearance_preferences_update_live(qapp):
    settings = QSettings("g4b3r-desktop", "JoeAudioStudio")
    settings.setValue("clipboard_dialog/font_family", "DejaVu Sans")
    settings.setValue("clipboard_dialog/phrase_font_size", 18)
    settings.setValue("clipboard_dialog/auxiliary_font_size", 15)
    settings.setValue("clipboard_dialog/title_font_size", 20)
    settings.setValue("clipboard_dialog/row_spacing", 6)
    settings.setValue("clipboard_dialog/window_padding", 24)
    settings.setValue("clipboard_dialog/accent", "violet")
    settings.sync()

    dialog = ClipboardAudioDialog(
        QMediaPlayer(),
        "",
        phrases=["Frase de aparência personalizada."],
    )
    dialog.apply_appearance_settings()

    assert dialog.appearance["font_family"] == "DejaVu Sans"
    assert dialog.appearance["phrase_font_size"] == 18
    assert dialog.appearance["auxiliary_font_size"] == 15
    assert dialog.appearance["title_font_size"] == 20
    assert dialog.appearance["row_spacing"] == 6
    assert dialog.appearance["window_padding"] == 24
    assert dialog.appearance["accent"] == "violet"
    assert dialog.phrase_list.font().family() == "DejaVu Sans"
    assert dialog.phrase_list.font().pixelSize() == 18
    assert dialog.phrase_list.spacing() == 6
    assert "#9a83e8" in dialog.styleSheet()

    dialog.close()
    settings.remove("clipboard_dialog")
    settings.sync()

