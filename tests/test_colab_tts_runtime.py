from __future__ import annotations

import numpy as np
import pytest

from tts_runtime import (
    configure_torch_threads,
    ensure_supported_torch,
    inspect_audio_samples,
    resolve_cpu_threads,
    resolve_tts_device,
)


def test_auto_prefers_cuda_when_available_and_cpu_otherwise():
    assert resolve_tts_device("auto", True) == "cuda"
    assert resolve_tts_device("auto", False) == "cpu"


def test_explicit_device_selection_is_respected():
    assert resolve_tts_device("cpu", True) == "cpu"
    assert resolve_tts_device("cuda", True) == "cuda"


def test_explicit_cuda_fails_when_runtime_has_no_cuda():
    with pytest.raises(RuntimeError, match="torch.cuda.is_available"):
        resolve_tts_device("cuda", False)


def test_invalid_device_mode_is_rejected():
    with pytest.raises(ValueError, match="JOE_TTS_DEVICE"):
        resolve_tts_device("gpu", True)


def test_cpu_threads_are_clamped_to_available_cpus():
    assert resolve_cpu_threads(8, cpu_count=2) == 2
    assert resolve_cpu_threads(1, cpu_count=2) == 1
    with pytest.raises(ValueError, match="pelo menos 1"):
        resolve_cpu_threads(0, cpu_count=2)


def test_configure_torch_threads_applies_intraop_and_interop_limits():
    class FakeTorch:
        intra = None
        inter = None

        @classmethod
        def set_num_threads(cls, count):
            cls.intra = count

        @classmethod
        def set_num_interop_threads(cls, count):
            cls.inter = count

    count = configure_torch_threads(FakeTorch, 1)

    assert count >= 1
    assert FakeTorch.intra == count
    assert FakeTorch.inter == 1


@pytest.mark.parametrize("version", ["2.5.0", "2.6.1+cu124", "2.10.0"])
def test_supported_pytorch_versions_are_accepted(version):
    assert ensure_supported_torch(version)[0] >= 2


@pytest.mark.parametrize("version", ["2.4.1", "2.3.0"])
def test_pytorch_24_or_older_is_rejected_with_audio_warning(version):
    with pytest.raises(RuntimeError, match="produz áudio incorreto"):
        ensure_supported_torch(version)


def test_audio_diagnostics_accept_normal_speech_signal():
    samples = np.sin(np.linspace(0, 30, 24_000, dtype=np.float32)) * 0.1

    result = inspect_audio_samples(samples, 24_000)

    assert result["duration_seconds"] == pytest.approx(1.0)
    assert result["peak"] == pytest.approx(0.1, rel=1e-3)
    assert result["rms"] > 0.05


def test_audio_diagnostics_reject_silence_and_non_finite_values():
    with pytest.raises(RuntimeError, match="silencioso"):
        inspect_audio_samples(np.zeros(24_000, dtype=np.float32), 24_000)
    broken = np.ones(24_000, dtype=np.float32)
    broken[100] = np.nan
    with pytest.raises(RuntimeError, match="NaN ou infinito"):
        inspect_audio_samples(broken, 24_000)
