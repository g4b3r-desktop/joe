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
        inference_mode=lambda: pytest.fail("benchmark must not enable inference_mode"),
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


def test_cuda_quantization_benchmark_excludes_warmups_and_measures_both_modes(
    monkeypatch,
):
    import sys
    from types import ModuleType, SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    loaded = []
    generated = {False: 0, True: 0}
    quantized_modules = []
    prompts = []

    class FakeAudio:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((1, 12_000), dtype=np.float32)

    class FakeModel:
        sample_rate = 24_000

        def __init__(self):
            self.flow_lm = SimpleNamespace(quantized=False)

        def to(self, device):
            assert device == "cuda"
            return self

        def get_state_for_audio_prompt(self, voice):
            prompts.append(voice)
            return object()

        def generate_audio(self, _state, _text, *, copy_state=True):
            assert copy_state is True
            generated[self.flow_lm.quantized] += 1
            return FakeAudio()

    class FakeTTSModel:
        @staticmethod
        def load_model(**kwargs):
            loaded.append(kwargs)
            return FakeModel()

    fake_pocket_tts = ModuleType("pocket_tts")
    fake_pocket_tts.TTSModel = FakeTTSModel
    fake_quantization = ModuleType("pocket_tts.quantization")
    fake_quantization.RECOMMENDED_CONFIG = {"attention", "ffn"}
    fake_quantization._get_backend = lambda: "torchao"

    def apply_dynamic_int8(flow_lm, groups):
        assert groups == {"attention", "ffn"}
        flow_lm.quantized = True
        quantized_modules.append(flow_lm)

    fake_quantization.apply_dynamic_int8 = apply_dynamic_int8
    fake_torch = SimpleNamespace(
        set_num_threads=lambda _value: None,
        inference_mode=lambda: pytest.fail("benchmark must not enable inference_mode"),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None,
            synchronize=lambda: None,
            max_memory_allocated=lambda: 256 * 1024 * 1024,
            max_memory_reserved=lambda: 384 * 1024 * 1024,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket_tts)
    monkeypatch.setitem(sys.modules, "pocket_tts.quantization", fake_quantization)

    results = PocketTTSService().benchmark_cuda_quantization(
        "Texto idêntico para os dois modos.",
        "portuguese",
        model_variant="portuguese",
        warmup_runs=2,
        measured_runs=3,
    )

    assert [result.quantized for result in results] == [False, True]
    assert all(not result.error for result in results)
    assert all(result.warmup_runs == 2 for result in results)
    assert all(len(result.generation_times) == 3 for result in results)
    assert all(len(result.audio_durations_seconds) == 3 for result in results)
    assert all(result.audio_durations_seconds == (0.5, 0.5, 0.5) for result in results)
    assert all(result.peak_vram_mb == 256 for result in results)
    assert all(result.peak_vram_reserved_mb == 384 for result in results)
    assert generated == {False: 5, True: 5}
    assert len(loaded) == 2
    assert all(kwargs["quantize"] is False for kwargs in loaded)
    assert len(quantized_modules) == 1
    assert len(prompts) == 2


def test_cuda_fp16_benchmark_wraps_each_pocket_tts_worker_in_autocast(monkeypatch):
    import sys
    from contextlib import contextmanager
    from types import ModuleType, SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    autocast_depth = {"value": 0}
    autocast_entries = []
    model_instances = []

    @contextmanager
    def fake_autocast(*, device_type, dtype):
        assert device_type == "cuda"
        assert dtype == "float16"
        autocast_depth["value"] += 1
        autocast_entries.append((device_type, dtype))
        try:
            yield
        finally:
            autocast_depth["value"] -= 1

    class FakeAudio:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.full((1, 12_000), 0.2, dtype=np.float32)

    class FakeModel:
        sample_rate = 24_000

        def __init__(self):
            self.worker_contexts = []
            model_instances.append(self)

        def to(self, device):
            assert device == "cuda"
            return self

        def get_state_for_audio_prompt(self, _voice):
            return object()

        def _generate(self):
            self.worker_contexts.append(autocast_depth["value"] > 0)

        def _autoregressive_generation(self):
            self.worker_contexts.append(autocast_depth["value"] > 0)

        def _decode_audio_worker(self):
            self.worker_contexts.append(autocast_depth["value"] > 0)

        def generate_audio(self, _state, _text, *, copy_state=True):
            assert copy_state is True
            self._generate()
            self._autoregressive_generation()
            self._decode_audio_worker()
            return FakeAudio()

    class FakeTTSModel:
        @staticmethod
        def load_model(**kwargs):
            assert kwargs["quantize"] is False
            return FakeModel()

    fake_pocket_tts = ModuleType("pocket_tts")
    fake_pocket_tts.TTSModel = FakeTTSModel
    fake_torch = SimpleNamespace(
        float16="float16",
        autocast=fake_autocast,
        set_num_threads=lambda _value: None,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None,
            synchronize=lambda: None,
            max_memory_allocated=lambda: 128 * 1024 * 1024,
            max_memory_reserved=lambda: 192 * 1024 * 1024,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket_tts)

    results = PocketTTSService().benchmark_cuda_precision(
        "Texto de teste para FP32 e FP16.",
        "portuguese",
        model_variant="portuguese",
        warmup_runs=2,
        measured_runs=3,
    )

    assert [result.precision for result in results] == ["FP32", "FP16 autocast"]
    assert all(not result.error for result in results)
    assert all(len(result.generation_times) == 3 for result in results)
    assert all(result.audio_durations_seconds == (0.5, 0.5, 0.5) for result in results)
    assert all(result.audio_peak_amplitudes == pytest.approx((0.2, 0.2, 0.2)) for result in results)
    assert all(result.audio_rms_levels == pytest.approx((0.2, 0.2, 0.2)) for result in results)
    assert len(model_instances) == 2
    assert all(context is False for context in model_instances[0].worker_contexts)
    assert all(context is True for context in model_instances[1].worker_contexts)
    assert len(model_instances[0].worker_contexts) == 15
    assert len(model_instances[1].worker_contexts) == 15
    assert len(autocast_entries) == 15
    assert autocast_depth["value"] == 0



def test_cuda_tf32_benchmark_switches_matmul_precision_and_restores_original(
    monkeypatch,
):
    import sys
    from types import ModuleType, SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    observed = []

    class FakeAudio:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.full((1, 12_000), 0.2, dtype=np.float32)

    matmul = SimpleNamespace(fp32_precision="ieee")

    class FakeModel:
        sample_rate = 24_000

        def to(self, device):
            assert device == "cuda"
            return self

        def get_state_for_audio_prompt(self, _voice):
            return object()

        def generate_audio(self, _state, _text, *, copy_state=True):
            assert copy_state is True
            observed.append(matmul.fp32_precision)
            return FakeAudio()

    class FakeTTSModel:
        @staticmethod
        def load_model(**kwargs):
            assert kwargs["quantize"] is False
            return FakeModel()

    fake_pocket_tts = ModuleType("pocket_tts")
    fake_pocket_tts.TTSModel = FakeTTSModel
    fake_torch = SimpleNamespace(
        set_num_threads=lambda _value: None,
        backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul)),
        cuda=SimpleNamespace(
            is_available=lambda: True,
            get_device_capability=lambda: (8, 6),
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None,
            synchronize=lambda: None,
            max_memory_allocated=lambda: 256 * 1024 * 1024,
            max_memory_reserved=lambda: 384 * 1024 * 1024,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket_tts)

    results = PocketTTSService().benchmark_cuda_tf32(
        "Texto idêntico para testar IEEE FP32 e TF32.",
        "portuguese",
        model_variant="portuguese",
        warmup_runs=2,
        measured_runs=3,
    )

    assert [result.precision for result in results] == ["FP32 IEEE", "FP32 TF32"]
    assert all(not result.error for result in results)
    assert all(len(result.generation_times) == 3 for result in results)
    assert all(result.audio_durations_seconds == (0.5, 0.5, 0.5) for result in results)
    assert observed == ["ieee"] * 5 + ["tf32"] * 5
    assert matmul.fp32_precision == "ieee"

def test_cuda_fp16_autocast_refuses_unknown_pocket_tts_thread_layout():
    from types import SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    class UnsupportedModel:
        def _generate(self):
            pass

    torch = SimpleNamespace(
        float16="float16",
        autocast=lambda **_kwargs: None,
    )

    with pytest.raises(RuntimeError, match="não expõe os pontos internos"):
        PocketTTSService._apply_cuda_fp16_autocast(UnsupportedModel(), torch)


def test_cuda_profile_measures_flow_mimi_and_latent_queue_on_worker_thread(monkeypatch):
    import sys
    import threading
    from types import ModuleType, SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    class FakeEvent:
        def __init__(self, enable_timing=False):
            assert enable_timing is True

        def record(self):
            pass

        def elapsed_time(self, _other):
            return 1.0

    class FakeAudio:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((1, 12_000), dtype=np.float32)

    class FakeTokens:
        shape = (1, 4)

    class FakeQueue:
        def get(self):
            return None

    class FakeMimi:
        def decode_from_latent(self, _latents, _state):
            return None

    class FakeTransformerLayer:
        def _sa_block(self, *_args, **_kwargs):
            return None

        def _ff_block(self, *_args, **_kwargs):
            return None

    class FakeFlowNet:
        def forward(self, *_args, **_kwargs):
            return None

    class FakeModel:
        sample_rate = 24_000

        def __init__(self):
            self.mimi = FakeMimi()
            self.flow_lm = SimpleNamespace(
                transformer=SimpleNamespace(layers=[FakeTransformerLayer()]),
                flow_net=FakeFlowNet(),
            )
            self.generation_calls = 0

        def to(self, device):
            assert device == "cuda"
            return self

        def get_state_for_audio_prompt(self, _voice):
            return object()

        def _run_flow_lm_and_increment_step(self, **kwargs):
            if kwargs.get("text_tokens") is None:
                layer = self.flow_lm.transformer.layers[0]
                layer._sa_block()
                layer._ff_block()
                self.flow_lm.flow_net.forward()
            return None

        def _decode_audio_worker(self, latents_queue, *_args):
            latents_queue.get()
            self.mimi.decode_from_latent(None, None)

        def generate_audio(self, _state, _text, *, copy_state=True):
            assert copy_state is True
            self.generation_calls += 1
            self._run_flow_lm_and_increment_step(text_tokens=FakeTokens())
            self._run_flow_lm_and_increment_step(backbone_input_latents=object())
            self._run_flow_lm_and_increment_step(backbone_input_latents=object())
            worker = threading.Thread(
                target=self._decode_audio_worker,
                args=(FakeQueue(), None, 0, 0),
            )
            worker.start()
            worker.join()
            return FakeAudio()

    models = []

    class FakeTTSModel:
        @staticmethod
        def load_model(**kwargs):
            assert kwargs["quantize"] is False
            model = FakeModel()
            models.append(model)
            return model

    fake_pocket_tts = ModuleType("pocket_tts")
    fake_pocket_tts.TTSModel = FakeTTSModel
    fake_torch = SimpleNamespace(
        set_num_threads=lambda _value: None,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None,
            synchronize=lambda: None,
            Event=FakeEvent,
            max_memory_allocated=lambda: 256 * 1024 * 1024,
            max_memory_reserved=lambda: 384 * 1024 * 1024,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket_tts)

    results = PocketTTSService().benchmark_cuda_profile(
        "Texto de diagnóstico por fases.",
        "portuguese",
        model_variant="portuguese",
        warmup_runs=2,
        measured_runs=3,
    )

    assert len(results) == 1
    result = results[0]
    assert not result.error
    assert len(result.generation_times) == 3
    assert result.audio_durations_seconds == (0.5, 0.5, 0.5)
    assert result.flow_step_counts == (2, 2, 2)
    assert result.flow_prefill_gpu_seconds == (0.001, 0.001, 0.001)
    assert result.flow_autoregressive_gpu_seconds == (0.002, 0.002, 0.002)
    assert result.flow_attention_gpu_seconds == (0.002, 0.002, 0.002)
    assert result.flow_ffn_gpu_seconds == (0.002, 0.002, 0.002)
    assert result.flow_net_gpu_seconds == (0.002, 0.002, 0.002)
    assert result.flow_attention_calls == (2, 2, 2)
    assert result.flow_ffn_calls == (2, 2, 2)
    assert result.flow_net_calls == (2, 2, 2)
    assert result.mimi_decode_calls == (1, 1, 1)
    assert result.mimi_decode_gpu_seconds == (0.001, 0.001, 0.001)
    assert result.latent_queue_get_calls == (1, 1, 1)
    assert result.peak_vram_mb == 256
    assert result.peak_vram_reserved_mb == 384
    assert models[0].generation_calls == 5


def test_cuda_profile_refuses_unknown_pocket_tts_internal_methods(monkeypatch):
    import sys
    from types import ModuleType, SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    class UnsupportedModel:
        sample_rate = 24_000

        def to(self, _device):
            return self

        def get_state_for_audio_prompt(self, _voice):
            return object()

    class FakeTTSModel:
        @staticmethod
        def load_model(**_kwargs):
            return UnsupportedModel()

    fake_pocket_tts = ModuleType("pocket_tts")
    fake_pocket_tts.TTSModel = FakeTTSModel
    fake_torch = SimpleNamespace(
        set_num_threads=lambda _value: None,
        cuda=SimpleNamespace(
            is_available=lambda: True,
            empty_cache=lambda: None,
            reset_peak_memory_stats=lambda: None,
            synchronize=lambda: None,
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket_tts)

    results = PocketTTSService().benchmark_cuda_profile(
        "Texto de teste.",
        "portuguese",
        model_variant="portuguese",
        warmup_runs=1,
        measured_runs=3,
    )

    assert len(results) == 1
    assert "não expõe os métodos internos" in results[0].error



def test_flash_fp16_attention_patch_casts_only_sdpa_and_restores(monkeypatch):
    import sys
    from contextlib import contextmanager
    from types import ModuleType, SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    backend_marker = object()
    entered_backends = []
    sdpa_calls = []

    @contextmanager
    def fake_sdpa_kernel(backend):
        entered_backends.append(backend)
        yield

    fake_torch = ModuleType("torch")
    fake_torch.__path__ = []
    fake_torch.float32 = "float32"
    fake_torch.float16 = "float16"

    fake_torch_nn = ModuleType("torch.nn")
    fake_torch_nn.__path__ = []
    fake_torch_attention = ModuleType("torch.nn.attention")

    class FakeSDPBackend:
        FLASH_ATTENTION = backend_marker
        EFFICIENT_ATTENTION = object()

    fake_torch_attention.SDPBackend = FakeSDPBackend
    fake_torch_attention.sdpa_kernel = fake_sdpa_kernel

    class FakeTensor:
        def __init__(self, dtype):
            self.dtype = dtype
            self.device = SimpleNamespace(type="cuda")
            self.shape = (1, 2, 3, 4)

        def stride(self):
            return (24, 12, 4, 1)

        def to(self, *, dtype):
            return FakeTensor(dtype)

    def original_sdpa(query, key, value, *_args, **_kwargs):
        sdpa_calls.append((query.dtype, key.dtype, value.dtype))
        return FakeTensor(query.dtype)

    original_functional = SimpleNamespace(
        scaled_dot_product_attention=original_sdpa
    )
    attention_module = ModuleType("pocket_tts.modules.attention")
    attention_module.F = original_functional

    pocket_tts = ModuleType("pocket_tts")
    pocket_tts.__path__ = []
    pocket_modules = ModuleType("pocket_tts.modules")
    pocket_modules.__path__ = []

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_torch_nn)
    monkeypatch.setitem(sys.modules, "torch.nn.attention", fake_torch_attention)
    monkeypatch.setitem(sys.modules, "pocket_tts", pocket_tts)
    monkeypatch.setitem(sys.modules, "pocket_tts.modules", pocket_modules)
    monkeypatch.setitem(
        sys.modules, "pocket_tts.modules.attention", attention_module
    )

    class FakeModel:
        def _generate(self):
            return self._autoregressive_generation()

        def _autoregressive_generation(self):
            tensor = FakeTensor("float32")
            return attention_module.F.scaled_dot_product_attention(
                tensor, tensor, tensor, dropout_p=0.0
            )

    model = FakeModel()
    restore = PocketTTSService._apply_cuda_flash_fp16_attention(model, fake_torch)

    result = model._generate()

    assert result.dtype == "float32"
    assert sdpa_calls == [("float16", "float16", "float16")]
    assert entered_backends == [backend_marker]

    restore()
    assert attention_module.F is original_functional
    assert model._generate() is not None
    assert len(entered_backends) == 1


def test_flash_fp16_attention_patch_refuses_missing_worker_methods(monkeypatch):
    from types import ModuleType, SimpleNamespace
    import sys

    from joe.services.pocket_tts import PocketTTSService

    fake_torch = ModuleType("torch")
    fake_torch.__path__ = []
    fake_attention_api = ModuleType("torch.nn.attention")
    fake_attention_api.SDPBackend = SimpleNamespace(FLASH_ATTENTION=object())

    class FakeContext:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake_attention_api.sdpa_kernel = lambda _backend: FakeContext()
    fake_torch_nn = ModuleType("torch.nn")
    fake_torch_nn.__path__ = []
    fake_module = ModuleType("pocket_tts.modules.attention")
    fake_module.F = SimpleNamespace(
        scaled_dot_product_attention=lambda *args, **kwargs: None
    )
    fake_pocket = ModuleType("pocket_tts")
    fake_pocket.__path__ = []
    fake_modules = ModuleType("pocket_tts.modules")
    fake_modules.__path__ = []

    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch.nn", fake_torch_nn)
    monkeypatch.setitem(sys.modules, "torch.nn.attention", fake_attention_api)
    monkeypatch.setitem(sys.modules, "pocket_tts", fake_pocket)
    monkeypatch.setitem(sys.modules, "pocket_tts.modules", fake_modules)
    monkeypatch.setitem(
        sys.modules, "pocket_tts.modules.attention", fake_module
    )

    class UnsupportedModel:
        pass

    with pytest.raises(RuntimeError, match="não expõe os pontos internos"):
        PocketTTSService._apply_cuda_flash_fp16_attention(
            UnsupportedModel(), fake_torch
        )


def test_cuda_attention_benchmark_runs_reference_and_flash_and_restores(monkeypatch):
    import sys
    from contextlib import contextmanager
    from types import ModuleType, SimpleNamespace

    from joe.services.pocket_tts import PocketTTSService

    backend_marker = object()
    entered_backends = []
    sdpa_dtypes = []
    seeds = []
    generated = []
    models = []

    @contextmanager
    def fake_sdpa_kernel(backend):
        entered_backends.append(backend)
        yield

    class FakeSDPBackend:
        FLASH_ATTENTION = backend_marker
        EFFICIENT_ATTENTION = object()

    class FakeTensor:
        def __init__(self, dtype):
            self.dtype = dtype
            self.device = SimpleNamespace(type="cuda")
            self.shape = (1, 2, 3, 4)

        def stride(self):
            return (24, 12, 4, 1)

        def to(self, *, dtype):
            return FakeTensor(dtype)

    def original_sdpa(query, key, value, *_args, **_kwargs):
        sdpa_dtypes.append((query.dtype, key.dtype, value.dtype))
        return FakeTensor(query.dtype)

    original_functional = SimpleNamespace(
        scaled_dot_product_attention=original_sdpa
    )
    attention_module = ModuleType("pocket_tts.modules.attention")
    attention_module.F = original_functional

    fake_torch = ModuleType("torch")
    fake_torch.__path__ = []
    fake_torch.float32 = "float32"
    fake_torch.float16 = "float16"
    fake_torch.set_num_threads = lambda _threads: None
    fake_torch.manual_seed = lambda seed: seeds.append(("cpu", seed))
    fake_torch.cuda = SimpleNamespace(
        is_available=lambda: True,
        empty_cache=lambda: None,
        reset_peak_memory_stats=lambda: None,
        synchronize=lambda: None,
        manual_seed_all=lambda seed: seeds.append(("cuda", seed)),
        max_memory_allocated=lambda: 256 * 1024 * 1024,
        max_memory_reserved=lambda: 384 * 1024 * 1024,
    )

    fake_torch_nn = ModuleType("torch.nn")
    fake_torch_nn.__path__ = []
    fake_torch_attention = ModuleType("torch.nn.attention")
    fake_torch_attention.SDPBackend = FakeSDPBackend
    fake_torch_attention.sdpa_kernel = fake_sdpa_kernel

    pocket_tts = ModuleType("pocket_tts")
    pocket_tts.__path__ = []
    pocket_modules = ModuleType("pocket_tts.modules")
    pocket_modules.__path__ = []

    class FakeAudio:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.zeros((1, 12_000), dtype=np.float32)

    class FakeModel:
        sample_rate = 24_000

        def __init__(self):
            self.generation_calls = 0
            models.append(self)

        def to(self, device):
            assert device == "cuda"
            return self

        def get_state_for_audio_prompt(self, _voice):
            return object()

        def _generate(self):
            return self._autoregressive_generation()

        def _autoregressive_generation(self):
            tensor = FakeTensor("float32")
            return attention_module.F.scaled_dot_product_attention(
                tensor, tensor, tensor, dropout_p=0.0
            )

        def generate_audio(self, _state, _text, *, copy_state=True):
            assert copy_state is True
            self.generation_calls += 1
            self._generate()
            generated.append(self)
            return FakeAudio()

    class FakeTTSModel:
        @staticmethod
        def load_model(**kwargs):
            assert kwargs["quantize"] is False
            return FakeModel()

    pocket_tts.TTSModel = FakeTTSModel

    for name, module in (
        ("torch", fake_torch),
        ("torch.nn", fake_torch_nn),
        ("torch.nn.attention", fake_torch_attention),
        ("pocket_tts", pocket_tts),
        ("pocket_tts.modules", pocket_modules),
        ("pocket_tts.modules.attention", attention_module),
    ):
        monkeypatch.setitem(sys.modules, name, module)

    monkeypatch.setattr(
        "joe.services.pocket_tts.configure_model_cache", lambda: None
    )

    results = PocketTTSService().benchmark_cuda_attention(
        "Texto idêntico nos dois modos.",
        "portuguese",
        model_variant="portuguese",
        warmup_runs=2,
        measured_runs=3,
    )

    assert [result.mode for result in results] == [
        "FP32 SDPA",
        "Flash FP16 SDPA",
        "Efficient FP16 SDPA",
    ]
    assert all(not result.error for result in results)
    assert all(len(result.generation_times) == 3 for result in results)
    assert all(result.audio_durations_seconds == (0.5, 0.5, 0.5) for result in results)
    assert all(result.audio_peak_amplitudes == (0.0, 0.0, 0.0) for result in results)
    assert all(result.audio_rms_levels == (0.0, 0.0, 0.0) for result in results)
    assert all(result.peak_vram_mb == 256 for result in results)
    assert all(result.peak_vram_reserved_mb == 384 for result in results)
    assert [model.generation_calls for model in models] == [5, 5, 5]
    assert sdpa_dtypes[:5] == [("float32",) * 3] * 5
    assert sdpa_dtypes[5:] == [("float16",) * 3] * 10
    assert len(entered_backends) == 10
    assert seeds == [
        (device, 1729 + index)
        for _mode in range(3)
        for index in range(5)
        for device in ("cpu", "cuda")
    ]
    assert attention_module.F is original_functional
