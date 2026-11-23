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
        ("Kassia_clip15s_pocket_tts.safetensors", ".safetensors"),
        ("/tmp/voice.safetensors", ".safetensors"),
        (r"C:\\voices\\voice.safetensors", ".safetensors"),
        ("reference.wav", ".wav"),
        ("reference.mp3", ".mp3"),
        ("reference.flac", ".flac"),
        (None, ".wav"),
        ("voice.invalid-extension-too-long", ".wav"),
    ],
)
def test_safe_upload_suffix_preserves_supported_upload_formats(filename, expected):
    assert safe_upload_suffix(filename) == expected


def test_safetensors_extension_is_not_rewritten_to_wav():
    suffix = safe_upload_suffix("Kassia_clip15s_pocket_tts.safetensors")

    assert suffix != ".wav"
    assert suffix == ".safetensors"
