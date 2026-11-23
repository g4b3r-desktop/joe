from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

COLAB_DIR = Path(__file__).resolve().parents[1] / "colab"
sys.path.insert(0, str(COLAB_DIR))

from drive_assets import (
    config_for_uploaded_model,
    model_fingerprint,
    replace_directory_atomically,
    resolve_voice_state,
    safe_asset_name,
    validate_model_files,
)


def _write_safetensors(path: Path) -> Path:
    payload = b"\x00" * 4
    header = json.dumps(
        {
            "tensor": {
                "dtype": "F32",
                "shape": [1],
                "data_offsets": [0, len(payload)],
            }
        }
    ).encode("utf-8")
    path.write_bytes(len(header).to_bytes(8, "little") + header + payload)
    return path


def test_safe_asset_name_is_a_single_portable_path_component():
    assert safe_asset_name("../../Kessya Voz 24C.wav") == "kessya-voz-24c"
    assert safe_asset_name("!!!") == "voice"


def test_model_fingerprint_changes_when_checkpoint_or_tokenizer_changes():
    base = {
        "model_variant": "portuguese_24l",
        "checkpoint_sha256": "a" * 64,
        "tokenizer_sha256": "b" * 64,
        "pocket_tts_version": "4.0.1",
    }
    changed = {**base, "checkpoint_sha256": "c" * 64}

    assert model_fingerprint(base) == model_fingerprint(dict(base))
    assert model_fingerprint(base) != model_fingerprint(changed)


def test_resolve_voice_state_rejects_other_model_fingerprint(tmp_path):
    root = tmp_path
    variant = "portuguese_24l"
    voice_id = "kessya-abc123"
    voice_dir = root / "voices" / variant / voice_id
    voice_dir.mkdir(parents=True)
    state_file = voice_dir / "voice_state.safetensors"
    state_file.write_bytes(b"state")
    (voice_dir / "metadata.json").write_text(
        json.dumps(
            {
                "voice_id": voice_id,
                "model_variant": variant,
                "model_fingerprint": "old-fingerprint",
                "voice_state_sha256": hashlib.sha256(state_file.read_bytes()).hexdigest(),
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outros pesos/tokenizer"):
        resolve_voice_state(root, variant, voice_id, "new-fingerprint")


def test_config_for_uploaded_model_uses_variant_template_and_drive_paths(tmp_path):
    import yaml

    config_dir = tmp_path / "templates"
    model_dir = tmp_path / "models" / "portuguese_24l"
    config_dir.mkdir()
    model_dir.mkdir(parents=True)
    template = {
        "weights_path": "hf://old/model.safetensors",
        "weights_path_without_voice_cloning": "hf://old/no-clone.safetensors",
        "flow_lm": {"lookup_table": {"tokenizer_path": "hf://old/tokenizer.json"}},
        "mimi": {"sample_rate": 24000},
    }
    (config_dir / "portuguese_24l.yaml").write_text(
        yaml.safe_dump(template), encoding="utf-8"
    )

    result = config_for_uploaded_model("portuguese_24l", model_dir, config_dir)

    assert result["weights_path"] == str((model_dir / "model.safetensors").resolve())
    assert result["weights_path_without_voice_cloning"] is None
    assert result["flow_lm"]["lookup_table"]["tokenizer_path"] == str(
        (model_dir / "tokenizer.json").resolve()
    )
    assert result["mimi"]["sample_rate"] == 24000


def test_validate_model_files_checks_safetensors_and_tokenizer(tmp_path):
    model = _write_safetensors(tmp_path / "model.safetensors")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text(json.dumps({"version": "1.0", "model": {}}), encoding="utf-8")

    result = validate_model_files(model, tokenizer)

    assert result["tensor_count"] == 1
    assert result["checkpoint_size"] == model.stat().st_size


def test_validate_model_files_rejects_invalid_tokenizer(tmp_path):
    model = _write_safetensors(tmp_path / "model.safetensors")
    tokenizer = tmp_path / "tokenizer.json"
    tokenizer.write_text("not-json", encoding="utf-8")

    with pytest.raises(json.JSONDecodeError):
        validate_model_files(model, tokenizer)


def test_replace_directory_atomically_replaces_previous_contents(tmp_path):
    destination = tmp_path / "models" / "portuguese"
    destination.mkdir(parents=True)
    (destination / "old.txt").write_text("old", encoding="utf-8")
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")

    replace_directory_atomically(staging, destination)

    assert not (destination / "old.txt").exists()
    assert (destination / "new.txt").read_text(encoding="utf-8") == "new"
    assert not staging.exists()
