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

    japanese_variants = model_variants_for_language("japanese")
    assert default_model_variant("japanese") == "japanese"
    assert [option[0] for option in japanese_variants] == ["japanese"]
    assert "experimental" in japanese_variants[0][1].lower()
    assert is_model_variant_for_language("japanese", "japanese")
    assert not is_model_variant_for_language("japanese", "japanese_24l")


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
        quantize=False,
    )

    assert loaded == [("portuguese_24l", False)]
    assert destination.read_bytes() == b"fake wav"

def test_generate_forwards_optional_model_and_chunk_parameters(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    loaded = []
    generated = []

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

        def generate_audio(self, _state, _text, **kwargs):
            generated.append(kwargs)
            return FakeAudio()

    class FakeTTSModel:
        @staticmethod
        def load_model(**kwargs):
            loaded.append(kwargs)
            return FakeModel()

    class FakeSoundFile:
        @staticmethod
        def write(path, _samples, _rate, subtype):
            assert subtype == "PCM_16"
            Path(path).write_bytes(b"fake wav")

        @staticmethod
        def info(_path):
            return SimpleNamespace(frames=3, samplerate=24_000)

    monkeypatch.setitem(
        sys.modules, "torch", SimpleNamespace(set_num_threads=lambda _n: None)
    )
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile)
    monkeypatch.setitem(
        sys.modules, "pocket_tts", SimpleNamespace(TTSModel=FakeTTSModel)
    )

    destination = tmp_path / "advanced.wav"
    PocketTTSService().generate(
        "Olá",
        "portuguese",
        destination,
        sampler_decode_steps=4,
        temperature=0.8,
        eos_threshold=-2.5,
        frames_after_eos=2,
        max_tokens=72,
        teacher_force_previous_chunk=True,
    )

    assert loaded == [
        {
            "language": "portuguese",
            "quantize": True,
            "sampler_decode_steps": 4,
            "temp": 0.8,
            "eos_threshold": -2.5,
        }
    ]
    assert generated == [
        {
            "frames_after_eos": 2,
            "max_tokens": 72,
            "teacher_force_previous_chunk": True,
        }
    ]
    assert destination.read_bytes() == b"fake wav"


def test_generate_japanese_uses_compact_config_and_japanese_voice(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from joe.pocket_tts_models import JAPANESE_DEFAULT_VOICE, JAPANESE_MODEL_CONFIG
    from joe.services.pocket_tts import PocketTTSService

    loaded = []
    prompts = []

    class FakeAudio:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return [0.0, 0.1, -0.1]

    class FakeModel:
        sample_rate = 24_000

        def get_state_for_audio_prompt(self, voice):
            prompts.append(voice)
            return object()

        def generate_audio(self, _state, _text):
            return FakeAudio()

    class FakeTTSModel:
        @staticmethod
        def load_model(**kwargs):
            loaded.append(kwargs)
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
    destination = tmp_path / "japanese.wav"
    service.generate("こんにちは。", "japanese", destination)

    assert loaded == [{"config": JAPANESE_MODEL_CONFIG, "quantize": True}]
    assert prompts == [JAPANESE_DEFAULT_VOICE]
    assert destination.read_bytes() == b"fake wav"



def test_generate_uses_selected_cuda_device_and_thread_count(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    loaded = []
    moved = []
    configured_threads = []

    class FakeAudio:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return [0.0, 0.1, -0.1]

    class FakeModel:
        sample_rate = 24_000

        def to(self, device):
            moved.append(device)
            return self

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

    fake_torch = SimpleNamespace(
        set_num_threads=lambda value: configured_threads.append(value),
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile)
    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(TTSModel=FakeTTSModel))

    service = PocketTTSService()
    destination = tmp_path / "cuda.wav"
    service.generate(
        "Olá",
        "portuguese",
        destination,
        model_variant="portuguese",
        device_preference="cuda",
        cpu_threads=4,
    )

    assert loaded == [("portuguese", False)]
    assert moved == ["cuda"]
    assert configured_threads == [4]
    assert destination.read_bytes() == b"fake wav"


def test_generate_cuda_fails_clearly_when_torch_cuda_is_unavailable(monkeypatch, tmp_path):
    import sys
    from types import SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    class FakeTTSModel:
        @staticmethod
        def load_model(**_kwargs):
            raise AssertionError("The model must not load when CUDA is unavailable.")

    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            set_num_threads=lambda _value: None,
            cuda=SimpleNamespace(is_available=lambda: False),
        ),
    )
    monkeypatch.setitem(sys.modules, "soundfile", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(TTSModel=FakeTTSModel))

    with pytest.raises(RuntimeError, match="CUDA.*não está disponível"):
        PocketTTSService().generate(
            "Olá",
            "portuguese",
            tmp_path / "cuda.wav",
            device_preference="cuda",
        )


def test_benchmark_reports_cpu_and_cuda_availability(monkeypatch):
    import sys
    from contextlib import nullcontext
    from types import SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    loaded = []

    class FakeAudio:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((1, 24_000), dtype=np.float32)

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

    fake_torch = SimpleNamespace(
        set_num_threads=lambda _value: None,
        inference_mode=nullcontext,
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pocket_tts", SimpleNamespace(TTSModel=FakeTTSModel))

    results = PocketTTSService().benchmark_devices(
        "Texto de teste",
        "portuguese",
        model_variant="portuguese",
        cpu_threads=3,
        repeats=2,
    )

    assert loaded == [("portuguese", True)]
    assert results[0].device == "cpu"
    assert results[0].error == ""
    assert results[0].cpu_threads == 3
    assert results[0].audio_duration_seconds == 1.0
    assert results[0].generation_seconds >= 0
    assert results[0].quantized
    assert results[1].device == "cuda"
    assert "CUDA não está disponível" in results[1].error


def test_generate_cuda_int8_is_opt_in_and_uses_torchao(monkeypatch, tmp_path):
    import sys
    from types import ModuleType, SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    quantized = []
    moved = []
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
        flow_lm = object()

        def to(self, device):
            moved.append(device)
            return self

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

    fake_pocket_tts = ModuleType("pocket_tts")
    fake_pocket_tts.TTSModel = FakeTTSModel
    fake_quantization = ModuleType("pocket_tts.quantization")
    fake_quantization.RECOMMENDED_CONFIG = {"attention", "ffn"}
    fake_quantization._get_backend = lambda: "torchao"
    fake_quantization.apply_dynamic_int8 = (
        lambda module, groups: quantized.append((module, groups))
    )

    fake_torch = SimpleNamespace(
        set_num_threads=lambda _value: None,
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile)
    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket_tts)
    monkeypatch.setitem(sys.modules, "pocket_tts.quantization", fake_quantization)

    destination = tmp_path / "cuda-int8.wav"
    PocketTTSService().generate(
        "Olá",
        "portuguese",
        destination,
        model_variant="portuguese",
        device_preference="cuda",
        quantize=True,
    )

    assert loaded == [("portuguese", False)]
    assert moved == ["cuda"]
    assert quantized == [(FakeModel.flow_lm, {"attention", "ffn"})]
    assert destination.read_bytes() == b"fake wav"


def test_generate_cuda_without_int8_does_not_import_or_apply_torchao(
    monkeypatch, tmp_path
):
    import sys
    from types import ModuleType, SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    loaded = []
    moved = []

    class FakeAudio:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return [0.0, 0.1, -0.1]

    class FakeModel:
        sample_rate = 24_000

        def to(self, device):
            moved.append(device)
            return self

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

    fake_pocket_tts = ModuleType("pocket_tts")
    fake_pocket_tts.TTSModel = FakeTTSModel
    fake_torch = SimpleNamespace(
        set_num_threads=lambda _value: None,
        cuda=SimpleNamespace(is_available=lambda: True, empty_cache=lambda: None),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "soundfile", FakeSoundFile)
    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket_tts)
    monkeypatch.delitem(sys.modules, "pocket_tts.quantization", raising=False)

    destination = tmp_path / "cuda-no-int8.wav"
    PocketTTSService().generate(
        "Olá",
        "portuguese",
        destination,
        model_variant="portuguese",
        device_preference="cuda",
        # None uses the safe device-specific default: no INT8 on CUDA.
    )

    assert loaded == [("portuguese", False)]
    assert moved == ["cuda"]
    assert "pocket_tts.quantization" not in sys.modules
    assert destination.read_bytes() == b"fake wav"
