from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

COLAB_DIR = Path(__file__).resolve().parents[1] / "colab"
sys.path.insert(0, str(COLAB_DIR))

from safetensors_validation import validate_safetensors_file


def _write_safetensors(path: Path, *, payload: bytes = b"\x00" * 4) -> Path:
    header = json.dumps(
        {
            "weight": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, len(payload)],
            }
        }
    ).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
    return path


def test_valid_safetensors_header_is_accepted(tmp_path):
    source = _write_safetensors(tmp_path / "voice.safetensors")

    result = validate_safetensors_file(source)

    assert result["tensor_count"] == 1
    assert result["header_size"] > 0
    assert result["file_size"] == source.stat().st_size


def test_wav_renamed_as_safetensors_is_identified(tmp_path):
    source = tmp_path / "voice.safetensors"
    source.write_bytes(b"RIFF" + (36).to_bytes(4, "little") + b"WAVEdata")

    with pytest.raises(ValueError, match="RIFF/WAVE"):
        validate_safetensors_file(source)


def test_zip_named_safetensors_is_identified(tmp_path):
    source = tmp_path / "voice.safetensors"
    source.write_bytes(b"PK\x03\x04" + b"\x00" * 32)

    with pytest.raises(ValueError, match="cabeçalho ZIP"):
        validate_safetensors_file(source)


def test_oversized_header_is_rejected_before_reading_it(tmp_path):
    source = tmp_path / "voice.safetensors"
    source.write_bytes((101 * 1024 * 1024).to_bytes(8, "little") + b"{}")

    with pytest.raises(ValueError, match="acima do limite"):
        validate_safetensors_file(source)


def test_truncated_header_is_identified(tmp_path):
    source = tmp_path / "voice.safetensors"
    source.write_bytes((128).to_bytes(8, "little") + b"{}")

    with pytest.raises(ValueError, match="cabeçalho truncado"):
        validate_safetensors_file(source)


def test_invalid_json_header_is_identified(tmp_path):
    source = tmp_path / "voice.safetensors"
    source.write_bytes((2).to_bytes(8, "little") + b"{}")

    # An empty JSON object has no tensor entries.
    with pytest.raises(ValueError, match="nenhum tensor"):
        validate_safetensors_file(source)


def test_tensor_offsets_must_fit_inside_file(tmp_path):
    source = tmp_path / "voice.safetensors"
    header = json.dumps(
        {
            "weight": {
                "dtype": "F32",
                "shape": [8],
                "data_offsets": [0, 64],
            }
        }
    ).encode("utf-8")
    source.write_bytes(len(header).to_bytes(8, "little") + header + b"\x00" * 4)

    with pytest.raises(ValueError, match="intervalo de dados inválido"):
        validate_safetensors_file(source)
