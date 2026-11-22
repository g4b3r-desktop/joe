from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from joe.services.pocket_tts import _write_wav_atomically


def test_wav_write_is_atomic_and_produces_a_valid_file(tmp_path):
    destination = tmp_path / "audio.wav"
    destination.write_bytes(b"old audio")
    samples = np.zeros(1600, dtype=np.float32)

    result = _write_wav_atomically(sf, samples, 16_000, destination)

    assert result == destination
    assert sf.info(destination).frames == 1600
    assert destination.read_bytes().startswith(b"RIFF")
    assert sorted(path.name for path in tmp_path.iterdir()) == ["audio.wav"]


def test_failed_wav_write_preserves_previous_audio_and_cleans_temp_file(tmp_path):
    destination = tmp_path / "audio.wav"
    destination.write_bytes(b"old audio")

    class FailingSoundFile:
        @staticmethod
        def write(path, *_args, **_kwargs):
            Path(path).write_bytes(b"partial")
            raise OSError("simulated write failure")

    with pytest.raises(OSError, match="simulated write failure"):
        _write_wav_atomically(
            FailingSoundFile(),
            np.zeros(10, dtype=np.float32),
            16_000,
            destination,
        )

    assert destination.read_bytes() == b"old audio"
    assert sorted(path.name for path in tmp_path.iterdir()) == ["audio.wav"]


def test_model_cache_is_inside_project_and_overrides_global_cache(monkeypatch, tmp_path):
    from joe.model_cache import configure_model_cache

    monkeypatch.setenv("HF_HOME", "/tmp/global-huggingface-cache")
    monkeypatch.setenv("HF_HUB_CACHE", "/tmp/global-hub-cache")

    cache_root = configure_model_cache(tmp_path)

    assert cache_root == tmp_path / ".models" / "huggingface"
    assert (cache_root / "hub").is_dir()
    assert os.environ["HF_HOME"] == str(cache_root)
    assert os.environ["HF_HUB_CACHE"] == str(cache_root / "hub")
    assert os.environ["HUGGINGFACE_HUB_CACHE"] == str(cache_root / "hub")


def test_model_catalog_exposes_variants_by_language():
    from joe.pocket_tts_models import (
        default_model_variant,
        is_model_variant_for_language,
        model_variants_for_language,
    )

    assert default_model_variant("portuguese") == "portuguese"
    assert is_model_variant_for_language("portuguese", "portuguese_24l")
    assert not is_model_variant_for_language("portuguese", "english_7m_2026-10")
    assert {option[0] for option in model_variants_for_language("english")} >= {
        "english",
        "english_18m_2026-10",
        "english_7m_refl_2026-10",
        "english_2m_2026-10",
        "english_drifting_26-09",
    }


def test_generate_uses_the_selected_model_variant(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    loaded = []

    class FakeAudio:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return [0.0, 0.1, -0.1]

    class FakeModel:
        sample_rate = 24_000

        def get_state_for_audio_prompt(self, _voice):
            return object()

        def generate_audio(self, _state, _text):
            return FakeAudio()

    class FakeTTSModel:
        @staticmethod
        def load_model(*, language, quantize):
            loaded.append((language, quantize))
            return FakeModel()

    class FakeSoundFile:
        @staticmethod
        def write(path, _samples, _rate, subtype):
            assert subtype == "PCM_16"
            Path(path).write_bytes(b"fake wav")

        @staticmethod
        def info(_path):
            return SimpleNamespace(frames=3, samplerate=24_000)

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _n: None))
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile)
    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(TTSModel=FakeTTSModel))

    service = PocketTTSService()
    destination = tmp_path / "variant.wav"
    service.generate(
        "Olá",
        "portuguese",
        destination,
        model_variant="portuguese_24l",
    )

    assert loaded == [("portuguese_24l", True)]
    assert destination.read_bytes() == b"fake wav"
