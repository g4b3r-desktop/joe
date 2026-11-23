from __future__ import annotations

import sys
from pathlib import Path

import pytest

COLAB_DIR = Path(__file__).resolve().parents[1] / "colab"
sys.path.insert(0, str(COLAB_DIR))

from upload_file_utils import safe_upload_suffix


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("reference.wav", ".wav"),
        ("reference.mp3", ".mp3"),
        ("reference.flac", ".flac"),
        (None, ".wav"),
        ("voice.invalid-extension-too-long", ".wav"),
        ("Kassia_clip15s_pocket_tts.safetensors", ".wav"),
    ],
)
def test_safe_upload_suffix_preserves_audio_formats_and_rejects_unknown_files(
    filename, expected
):
    assert safe_upload_suffix(filename) == expected


@pytest.mark.parametrize(
    "filename",
    [
        "Kassia_clip15s_pocket_tts.safetensors",
        "/tmp/voice.safetensors",
        r"C:\\voices\\voice.safetensors",
    ],
)
def test_tts_voice_upload_preserves_safetensors_extension(filename):
    assert safe_upload_suffix(filename, allow_safetensors=True) == ".safetensors"
